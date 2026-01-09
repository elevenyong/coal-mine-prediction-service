"""
煤矿瓦斯风险预测系统 - 数据预处理模块（最终干净版 DataPreprocessor）

目标：
- 只保留一条清晰主路径：DataFrame构造 -> 列名标准化 -> 基础字段处理 -> 时空增强 -> 特征对齐/输出
- 训练阶段：history seed 注入（唯一入口）-> 历史统计/趋势 -> 剔除 seed 行 ->（可选）bridge
- 预测阶段：缺失增强特征先补0 -> 从 t_feature_cache 回填一次（避免线上长期全0导致分布漂移）
- 强防御：任何异常不影响主流程（尽量回退/补默认），不破坏核心功能接口

注意：
- 该类保留了你现有项目依赖的所有方法名/签名：
  __init__ / _load_feature_config / preprocess_data / _add_spatiotemporal_identifier /
  _enrich_spatiotemporal_features / _drop_history_seed_rows / _add_distance_to_face_bucket /
  _ensure_spatiotemporal_group / _generate_spatiotemporal_features / _fill_enhanced_from_feature_cache /
  _bridge_training_history_from_feature_cache / _sanitize_numeric_values / _log_data_quality_summary /
  _seed_training_history_from_db
"""

import pandas as pd
import numpy as np
from loguru import logger

from config_utils import ConfigUtils


class DataPreprocessor(ConfigUtils):
    """数据预处理器（最终干净版）"""

    def __init__(self, config_path="config.ini"):
        super().__init__(config_path)
        self._load_feature_config()

        # -----------------------------------------------------------------
        # DB能力挂载（由 preprocess_data(db_utils=...) 注入）
        #   - _db_utils_for_train：训练阶段 bridge 需要用到的DB能力
        #   - _db_utils_for_enrich：训练阶段 history seed 注入需要用到的DB能力
        #   - _db_utils_for_predict：预测阶段增强特征回填需要用到的DB能力
        #   - _is_training：标记当前 preprocess 调用是否训练模式
        # -----------------------------------------------------------------
        self._db_utils_for_train = None
        self._db_utils_for_enrich = None
        self._db_utils_for_predict = None
        self._is_training = False

    def _load_feature_config(self):
        """
        私有方法：加载特征配置（[Features] section）
        """
        try:
            categorical_str = self.config.get("Features", "base_categorical", fallback="")
            self.base_categorical = [x.strip() for x in categorical_str.split(",") if x.strip()]

            numeric_str = self.config.get("Features", "base_numeric", fallback="")
            self.base_numeric = [x.strip() for x in numeric_str.split(",") if x.strip()]

            target_str = self.config.get("Features", "target_features", fallback="")
            self.target_features = [x.strip() for x in target_str.split(",") if x.strip()]

            if not self.base_numeric:
                logger.warning("未配置基础数值特征（base_numeric），模型训练可能失败")
            if not self.target_features:
                raise ValueError("必须配置至少一个预测目标特征（target_features）")

            logger.debug(
                f"特征配置加载完成：分类={self.base_categorical}，数值={self.base_numeric}，目标={self.target_features}"
            )
        except Exception as e:
            logger.error(f"加载特征配置失败：{repr(e)}", exc_info=True)
            raise

    def preprocess_data(self, data, is_training=False, fault_calculator=None, db_utils=None):
        """
        主入口：预处理数据并产出训练/预测可用的特征DataFrame。

        :param data: list[dict] 或 dict 或 pd.DataFrame
        :param is_training: bool，训练=True；预测=False
        :param fault_calculator: 断层影响计算器（可选）
        :param db_utils: DB工具（可选）
            - 若传入单个对象：同时挂到 train/enrich/predict
            - 若传入 dict：可用键 train/enrich/predict 分别挂载
        :return:
            - 训练：return (df, feature_cols)
            - 预测：return df
        """
        self._is_training = bool(is_training)

        # -------------------- 0) DB挂载（强防御） --------------------
        try:
            if isinstance(db_utils, dict):
                self._db_utils_for_train = db_utils.get("train", None)
                self._db_utils_for_enrich = db_utils.get("enrich", None)
                self._db_utils_for_predict = db_utils.get("predict", None)
            else:
                # 单对象：默认三者同源
                self._db_utils_for_train = db_utils
                self._db_utils_for_enrich = db_utils
                self._db_utils_for_predict = db_utils
        except Exception:
            self._db_utils_for_train = None
            self._db_utils_for_enrich = None
            self._db_utils_for_predict = None

        # -------------------- 1) 构造DataFrame --------------------
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, (list, tuple)):
            df = pd.DataFrame(list(data))
        else:
            raise ValueError("data 必须为 DataFrame / dict / list[dict]")

        if df is None or len(df) == 0:
            # 空数据直接返回
            return (df, []) if self._is_training else df

        # -------------------- 2) 列名标准化（必须最前） --------------------
        try:
            df.columns = (
                df.columns.astype(str)
                .str.strip()
                .str.replace(" ", "_")
                .str.replace("-", "_")
            )
        except Exception:
            pass

        # -------------------- 3) measurement_date 解析（天级） --------------------
        if "measurement_date_parsed" not in df.columns and "measurement_date" in df.columns:
            try:
                df["measurement_date_parsed"] = pd.to_datetime(df["measurement_date"], errors="coerce")
            except Exception:
                pass

        # -------------------- 4) 断层影响（可选） --------------------
        # 说明：fault_calculator 若提供，尽量产出 fault_influence_strength；失败不影响主流程
        if fault_calculator is not None:
            try:
                if "fault_influence_strength" not in df.columns:
                    # 兼容多种计算器接口：calculate/compute 或 callable
                    if hasattr(fault_calculator, "calculate"):
                        df["fault_influence_strength"] = fault_calculator.calculate(df)
                    elif hasattr(fault_calculator, "compute"):
                        df["fault_influence_strength"] = fault_calculator.compute(df)
                    elif callable(fault_calculator):
                        df["fault_influence_strength"] = fault_calculator(df)
            except Exception as e:
                logger.warning(f"断层影响特征计算失败（将跳过）：{repr(e)}", exc_info=True)

        # -------------------- 5) 添加时空唯一标识（用于避免错误去重） --------------------
        try:
            df = self._add_spatiotemporal_identifier(df)
        except Exception:
            pass

        # -------------------- 6) 补充关键时空特征（包含 face_advance_distance 自动计算） --------------------
        try:
            df = self._enrich_spatiotemporal_features(df)
        except Exception as e:
            logger.warning(f"补充关键时空特征失败（将跳过）：{repr(e)}", exc_info=True)

        # -------------------- 7) 生成时空交互/历史趋势特征（核心） --------------------
        try:
            df = self._generate_spatiotemporal_features(df)
        except Exception as e:
            logger.warning(f"生成时空交互特征失败（将跳过）：{repr(e)}", exc_info=True)

        # -------------------- 8) 数值安全清洗（inf/NaN -> 0） --------------------
        try:
            df = self._sanitize_numeric_values(df)
        except Exception:
            pass

        # -------------------- 9) 输出列选择（训练返回 df+feature_cols；预测返回 df） --------------------
        if self._is_training:
            # 训练必须包含目标列
            missing_targets = [t for t in self.target_features if t not in df.columns]
            if missing_targets:
                raise ValueError(f"训练数据缺少目标特征：{missing_targets}")

            # 训练特征列：按配置 base_numeric/base_categorical + 自动增强（不含目标列/日期列/内部列）
            feature_cols = []
            # 先按配置顺序收集
            for c in (self.base_categorical + self.base_numeric):
                if c in df.columns and c not in feature_cols:
                    feature_cols.append(c)

            # 自动增强列：非下划线开头，且不属于目标列、日期列
            blocked = set(self.target_features) | {
                "measurement_date", "measurement_date_parsed",
                "_spatiotemporal_id", "_tmp_dt_for_sort"
            }
            for c in df.columns:
                if c.startswith("_"):
                    continue
                if c in blocked:
                    continue
                # 排除明显的日期字符串列（防止脏日期混入特征）
                if df[c].dtype == "object":
                    # 粗过滤：形如 YYYY-MM-DD 的字符串列不作为特征
                    try:
                        sample = df[c].dropna().astype(str).head(3).tolist()
                        if any(len(x) >= 8 and x[0:4].isdigit() and "-" in x for x in sample):
                            continue
                    except Exception:
                        pass
                if c not in feature_cols and c in df.columns:
                    feature_cols.append(c)

            # 确保特征列不包含目标列
            feature_cols = [c for c in feature_cols if c not in self.target_features]

            logger.debug(f"训练特征确定：共 {len(feature_cols)} 个")
            self._log_data_quality_summary(df)
            return df, feature_cols

        # 预测：严格按训练特征顺序输出（由调用方传入 keep_cols 时在外部对齐；这里尽量保持所有列）
        return df

    def _add_spatiotemporal_identifier(self, df):
        """
        添加时空唯一标识，避免错误去重
        """
        identifier_parts = []
        for col in ["working_face", "x_coord", "y_coord", "z_coord"]:
            if col in df.columns:
                identifier_parts.append(col)
        if "measurement_date" in df.columns:
            identifier_parts.append("measurement_date")
        if "borehole_id" in df.columns:
            identifier_parts.append("borehole_id")
        if "drilling_depth" in df.columns:
            identifier_parts.append("drilling_depth")
        elif "face_advance_distance" in df.columns:
            identifier_parts.append("face_advance_distance")

        if identifier_parts:
            available = [c for c in identifier_parts if c in df.columns]
            if available:
                try:
                    df["_spatiotemporal_id"] = df[available].astype(str).agg("_".join, axis=1)
                except Exception:
                    df["_spatiotemporal_id"] = df.index.astype(str)
            else:
                df["_spatiotemporal_id"] = df.index.astype(str)
        else:
            df["_spatiotemporal_id"] = df.index.astype(str)

        return df

    def _enrich_spatiotemporal_features(self, df):
        """
        补充关键时空特征（含 face_advance_distance 自动计算）
        """
        # 0) 自动计算 face_advance_distance（如缺失）
        if "face_advance_distance" not in df.columns:
            df["face_advance_distance"] = 0.0

            try:
                # 确保解析日期
                if "measurement_date_parsed" not in df.columns and "measurement_date" in df.columns:
                    df["measurement_date_parsed"] = pd.to_datetime(df["measurement_date"], errors="coerce")

                # 分组键：优先 workface_id/working_face
                group_cols = []
                if "workface_id" in df.columns:
                    group_cols.append("workface_id")
                if "working_face" in df.columns:
                    group_cols.append("working_face")
                if not group_cols:
                    df["__all__"] = 0
                    group_cols = ["__all__"]

                work_stage_series = df["work_stage"].astype(str) if "work_stage" in df.columns else pd.Series([""] * len(df))
                roadway_series = df["roadway"].astype(str) if "roadway" in df.columns else pd.Series([""] * len(df))
                is_mining_mask = work_stage_series.str.contains("回采", na=False) | (roadway_series.str.lower() == "mining")

                # 回采：distance_to_face 递减，推进量 = prev - current
                # 掘进：distance_from_entrance 递增，推进量 = current - prev
                # 若两列都无，则保持 0
                if "measurement_date_parsed" in df.columns:
                    # 用原index回填，避免乱序
                    adv_series = pd.Series(0.0, index=df.index)

                    for _, g in df.groupby(group_cols):
                        g2 = g.sort_values("measurement_date_parsed")
                        if "distance_to_face" in g2.columns and is_mining_mask.loc[g2.index].any():
                            base = g2["distance_to_face"]
                            # prev - curr
                            adv = base.shift(1) - base
                        elif "distance_from_entrance" in g2.columns:
                            base = g2["distance_from_entrance"]
                            # curr - prev
                            adv = base - base.shift(1)
                        else:
                            adv = pd.Series(0.0, index=g2.index)

                        adv = adv.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                        # 推进量不应为负（出现负数大概率是异常噪声/口径反）
                        try:
                            adv = adv.where(adv >= 0, 0.0)
                        except Exception:
                            pass

                        adv_series.loc[g2.index] = adv

                    df["face_advance_distance"] = adv_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # 清理临时列
                if "__all__" in df.columns:
                    df = df.drop(columns=["__all__"])

            except Exception as e:
                logger.warning(f"自动计算 face_advance_distance 失败（将回退为0）：{repr(e)}", exc_info=True)
                df["face_advance_distance"] = 0.0

        # 1) days_since_start / days_in_workface 若不存在，可在此补0（真正计算口径可后续继续迭代）
        for c in ["days_since_start", "days_in_workface"]:
            if c not in df.columns:
                df[c] = 0.0

        return df

    def _drop_history_seed_rows(self, df):
        """
        训练阶段：剔除 history seed 行
        规则：_is_history_seed == 1 或 True 的行全部剔除
        """
        try:
            if df is None or len(df) == 0:
                return df
            if "_is_history_seed" not in df.columns:
                return df
            s = df["_is_history_seed"]
            # 兼容 bool / int / str
            try:
                mask = s.fillna(0).astype(int) == 1
            except Exception:
                mask = s.fillna(False).astype(bool)
            if mask.any():
                df = df.loc[~mask].copy()
            # 不把内部列带出去
            try:
                df = df.drop(columns=["_is_history_seed"])
            except Exception:
                pass
            return df
        except Exception:
            return df

    def _add_distance_to_face_bucket(self, df):
        """
        距采面距离分桶（统一口径，避免两套bucket导致 group 不一致）
        """
        try:
            if "distance_to_face" not in df.columns:
                return df
            bucket = self.config.getfloat("SpatioTemporal", "distance_to_face_bucket", fallback=2.0)
            if bucket <= 0:
                return df
            df["distance_to_face_bucket"] = (df["distance_to_face"].astype(float) / float(bucket)).round(0) * float(bucket)
            df["distance_to_face_bucket"] = df["distance_to_face_bucket"].round(2)
        except Exception:
            pass
        return df

    def _ensure_spatiotemporal_group(self, df):
        """
        防御：保证 spatiotemporal_group 至少有一个可用兜底值
        """
        try:
            if df is None or len(df) == 0:
                return df
            if "spatiotemporal_group" not in df.columns:
                # 尽量用 coord_hash，否则用 workface_id，否则用 default
                if "coord_hash" in df.columns:
                    df["spatiotemporal_group"] = df["coord_hash"].astype(str)
                elif "workface_id" in df.columns:
                    df["spatiotemporal_group"] = df["workface_id"].astype(str)
                else:
                    df["spatiotemporal_group"] = "default_group"

            # 空值兜底
            df["spatiotemporal_group"] = df["spatiotemporal_group"].astype(str).replace("nan", "").fillna("")
            df.loc[df["spatiotemporal_group"].str.strip() == "", "spatiotemporal_group"] = "default_group"
        except Exception:
            pass
        return df

    def _generate_spatiotemporal_features(self, df):
        """
        生成时空交互特征（最终干净版主路径）
        """
        # 0) 防御：确保 group 至少可用
        df = self._ensure_spatiotemporal_group(df)

        # 1) 空间-时间交互项
        if "distance_to_face" in df.columns and "days_since_start" in df.columns:
            try:
                df["distance_time_interaction"] = (df["distance_to_face"].astype(float) * df["days_since_start"].astype(float)) / 1000.0
                df["distance_time_interaction"] = df["distance_time_interaction"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            except Exception:
                df["distance_time_interaction"] = 0.0

        # 2) 推进速率 advance_rate（如可计算）
        if "face_advance_distance" in df.columns and "measurement_date_parsed" in df.columns and "working_face" in df.columns:
            try:
                adv_rate = pd.Series(0.0, index=df.index)
                for wf, g in df.groupby("working_face"):
                    g2 = g.sort_values("measurement_date_parsed")
                    delta_days = g2["measurement_date_parsed"].diff().dt.total_seconds() / 86400.0
                    delta_days = delta_days.replace(0, np.nan).where(delta_days > 0)
                    rate = g2["face_advance_distance"].diff() / delta_days
                    rate = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    adv_rate.loc[g2.index] = rate
                df["advance_rate"] = adv_rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            except Exception:
                df["advance_rate"] = 0.0

        # 3) 历史统计/趋势：构造 coord_hash +（可选）distance_to_face_bucket
        if all(c in df.columns for c in ["x_coord", "y_coord", "z_coord"]):
            try:
                coord_round = self.config.getint("SpatioTemporal", "coord_round", fallback=1)
                history_group_mode = self.config.get("SpatioTemporal", "history_group_mode", fallback="coord_distance")

                # coord_hash
                df["coord_hash"] = (
                    df["x_coord"].round(coord_round).astype(str) + "_" +
                    df["y_coord"].round(coord_round).astype(str) + "_" +
                    df["z_coord"].round(coord_round).astype(str)
                )

                # distance_to_face_bucket
                df = self._add_distance_to_face_bucket(df)

                # spatiotemporal_group
                if history_group_mode == "coord_distance" and "distance_to_face_bucket" in df.columns:
                    df["spatiotemporal_group"] = df["coord_hash"].astype(str) + "_" + df["distance_to_face_bucket"].astype(str)
                else:
                    df["spatiotemporal_group"] = df["coord_hash"].astype(str)

                df = self._ensure_spatiotemporal_group(df)
                group_key = "spatiotemporal_group"

                # 确保 parsed 日期
                if "measurement_date_parsed" not in df.columns and "measurement_date" in df.columns:
                    df["measurement_date_parsed"] = pd.to_datetime(df["measurement_date"], errors="coerce")

                # ------------------------------
                # 训练阶段：history seed 注入（唯一入口）
                # ------------------------------
                if getattr(self, "_is_training", False):
                    df, seed_added = self._seed_training_history_from_db(df)
                    try:
                        logger.info(f"训练阶段历史种子接续完成：注入 {int(seed_added)} 条 seed 行")
                    except Exception:
                        pass

                # 排序：workface_id + group + date
                try:
                    if "measurement_date_parsed" in df.columns:
                        if "workface_id" in df.columns:
                            df = df.sort_values(["workface_id", group_key, "measurement_date_parsed"], kind="mergesort")
                        else:
                            df = df.sort_values([group_key, "measurement_date_parsed"], kind="mergesort")
                except Exception:
                    pass

                # 历史均值/趋势
                for target in ["gas_emission_q", "drilling_cuttings_s", "gas_emission_velocity_q"]:
                    if target in df.columns:
                        historical_mean = (
                            df.groupby(group_key)[target]
                            .shift(1)
                            .expanding()
                            .mean()
                            .reset_index(level=0, drop=True)
                        )
                        df[f"{target}_historical_mean"] = historical_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                        trend = df.groupby(group_key)[target].diff()
                        df[f"{target}_trend"] = trend.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    else:
                        # 预测阶段：先补0，后续从 feature_cache 回填
                        df[f"{target}_historical_mean"] = 0.0
                        df[f"{target}_trend"] = 0.0

                # ------------------------------
                # 训练/预测分支收敛
                # ------------------------------
                if getattr(self, "_is_training", False):
                    # 训练：剔除 seed 行
                    df = self._drop_history_seed_rows(df)

                    # 训练：bridge（可选）
                    try:
                        if self._db_utils_for_train is not None:
                            self._bridge_training_history_from_feature_cache(df, db_utils=self._db_utils_for_train, group_key=group_key)
                    except Exception as e:
                        logger.warning(f"训练阶段bridge失败（将跳过）：{repr(e)}", exc_info=True)

                else:
                    # 预测：回填增强特征（只回填一次）
                    try:
                        if self._db_utils_for_predict is not None:
                            self._fill_enhanced_from_feature_cache(df, db_utils=self._db_utils_for_predict)
                    except Exception as e:
                        logger.warning(f"预测阶段增强特征回填失败（将跳过）：{repr(e)}", exc_info=True)

            except Exception as e:
                logger.warning(f"时空增强历史统计生成失败（将回退为基础输出）：{repr(e)}", exc_info=True)

        return df

    def _fill_enhanced_from_feature_cache(self, df, db_utils):
        """
        预测阶段：从 t_feature_cache 回填增强特征（historical_mean/trend/advance_rate等）
        db_utils 需要实现类似方法：get_feature_cache_by_keys(...) 或等价接口
        """
        if df is None or len(df) == 0 or db_utils is None:
            return df

        # 允许不同DB工具实现：只要存在一个可用的方法就调用
        getter = None
        for cand in ["get_feature_cache_by_keys", "get_feature_cache", "get_latest_feature_cache"]:
            if hasattr(db_utils, cand):
                getter = getattr(db_utils, cand)
                break
        if getter is None:
            return df

        # 需要的键：workface_id + spatiotemporal_group + measurement_date（可选）
        key_cols = ["workface_id", "spatiotemporal_group"]
        if not all(c in df.columns for c in key_cols):
            return df

        # 逐行回填（数据量不大时可接受；若后续需要可再批量优化）
        try:
            for idx, row in df.iterrows():
                try:
                    wid = int(float(row.get("workface_id")))
                    grp = str(row.get("spatiotemporal_group", "")).strip()
                    if not grp:
                        continue

                    # 支持带日期的接口
                    mdt = None
                    if "measurement_date" in df.columns:
                        mdt = str(row.get("measurement_date", "")).strip() or None

                    try:
                        if mdt is not None:
                            cached = getter(workface_id=wid, spatiotemporal_group=grp, measurement_date=mdt)
                        else:
                            cached = getter(workface_id=wid, spatiotemporal_group=grp)
                    except TypeError:
                        # 兼容不同签名
                        cached = getter(wid, grp)

                    if not cached or not isinstance(cached, dict):
                        continue

                    for k, v in cached.items():
                        # 只回填增强列，不覆盖基础输入列
                        if k in df.columns and (k.endswith("_historical_mean") or k.endswith("_trend") or k in ["advance_rate"]):
                            try:
                                if pd.isna(df.at[idx, k]) or float(df.at[idx, k]) == 0.0:
                                    df.at[idx, k] = v
                            except Exception:
                                pass

                except Exception:
                    continue
        except Exception:
            pass

        return df

    def _bridge_training_history_from_feature_cache(self, df, db_utils, group_key):
        """
        训练阶段：bridge 接续（仅修补“组内首条记录”的历史/趋势退化问题）
        db_utils 需要实现类似方法：get_latest_enhanced_by_group(...) 或等价接口
        """
        if df is None or len(df) == 0 or db_utils is None:
            return df
        if group_key not in df.columns:
            return df

        getter = None
        for cand in ["get_latest_enhanced_by_group", "get_latest_feature_cache", "get_feature_cache"]:
            if hasattr(db_utils, cand):
                getter = getattr(db_utils, cand)
                break
        if getter is None:
            return df

        # 对每个 group 找本批首条记录（按 measurement_date_parsed）
        if "measurement_date_parsed" in df.columns:
            groups = df.groupby(group_key, sort=False)
            for g, sub in groups:
                try:
                    first_idx = sub.sort_values("measurement_date_parsed").index[0]
                except Exception:
                    first_idx = sub.index[0]

                try:
                    wid = int(float(df.at[first_idx, "workface_id"])) if "workface_id" in df.columns else None
                    grp = str(df.at[first_idx, group_key])
                    if wid is None or not grp:
                        continue

                    try:
                        cached = getter(workface_id=wid, spatiotemporal_group=grp)
                    except TypeError:
                        cached = getter(wid, grp)

                    if not cached or not isinstance(cached, dict):
                        continue

                    # 仅在当前为0时补上上一条增强值
                    for k, v in cached.items():
                        if k in df.columns and (k.endswith("_historical_mean") or k.endswith("_trend") or k in ["advance_rate"]):
                            try:
                                cur = df.at[first_idx, k]
                                if pd.isna(cur) or float(cur) == 0.0:
                                    df.at[first_idx, k] = v
                            except Exception:
                                pass

                except Exception:
                    continue

        return df

    def _sanitize_numeric_values(self, df):
        """
        数值安全清洗：inf/NaN -> 0（只处理数值列）
        """
        if df is None or len(df) == 0:
            return df
        try:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                return df
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            pass
        return df

    def _log_data_quality_summary(self, df):
        """
        训练阶段：输出关键列质量概览（用于快速诊断“趋势/历史为0过多”等问题）
        """
        try:
            if df is None or len(df) == 0:
                return
            cols = [
                "spatiotemporal_group",
                "drilling_cuttings_s_historical_mean",
                "gas_emission_velocity_q_historical_mean",
                "gas_emission_q_historical_mean",
                "drilling_cuttings_s_trend",
                "gas_emission_velocity_q_trend",
                "gas_emission_q_trend",
                "advance_rate"
            ]
            exist = [c for c in cols if c in df.columns]
            if not exist:
                return

            logger.info("【数据质量概览】增强特征缺失/为0比例：")
            for c in exist:
                try:
                    s = df[c]
                    if s.dtype == "object":
                        continue
                    zero_ratio = float((s.fillna(0.0) == 0.0).mean())
                    nan_ratio = float(s.isna().mean())
                    logger.info(f"  - {c}: NaN占比={nan_ratio:.2%}, 0占比={zero_ratio:.2%}")
                except Exception:
                    continue
        except Exception:
            pass

    def _seed_training_history_from_db(self, df: pd.DataFrame, seed_limit: int = None):
        """
        训练阶段：从DB取“当前批次最早日期之前”的历史真实目标值，拼接到df前面作为历史统计种子。

        强约束：
          1) 任何异常都不得影响主流程：本函数永不 raise
          2) 返回值稳定：始终返回 (df_out, seed_rows_count)
          3) 强防御：DB方法缺失/返回None/字段缺失/类型不对都可容忍
          4) 宁可少seed也不允许失败
        """
        try:
            if df is None or len(df) == 0:
                return df, 0
            if not bool(getattr(self, "_is_training", False)):
                return df, 0

            dbu = getattr(self, "_db_utils_for_enrich", None)
            if dbu is None or (not hasattr(dbu, "get_recent_targets_by_group")):
                return df, 0

            need_cols = ["workface_id", "spatiotemporal_group", "measurement_date"]
            for c in need_cols:
                if c not in df.columns:
                    return df, 0

            if seed_limit is None:
                try:
                    seed_limit = self.config.getint("SpatioTemporal", "history_seed_limit", fallback=200)
                except Exception:
                    seed_limit = 200

            try:
                seed_limit = int(seed_limit)
                if seed_limit <= 0:
                    return df, 0
            except Exception:
                seed_limit = 200

            # 每个组的最早日期
            try:
                min_dates = (
                    df.groupby(["workface_id", "spatiotemporal_group"])["measurement_date"]
                    .min()
                    .reset_index()
                )
            except Exception:
                return df, 0

            if min_dates is None or len(min_dates) == 0:
                return df, 0

            # 限流：避免组太多时查库过猛
            try:
                if len(min_dates) > 5000:
                    min_dates = min_dates.head(5000)
            except Exception:
                pass

            seed_rows = []
            seed_flag_col = "_is_history_seed"

            qcache = {}
            for _, r in min_dates.iterrows():
                try:
                    wid_raw = r.get("workface_id", None)
                    grp_raw = r.get("spatiotemporal_group", None)
                    mdt_raw = r.get("measurement_date", None)
                    if wid_raw is None or grp_raw is None or mdt_raw is None:
                        continue

                    try:
                        wid = int(float(wid_raw))
                    except Exception:
                        continue

                    grp = str(grp_raw).strip()
                    mdt = str(mdt_raw).strip()
                    if not grp or not mdt:
                        continue

                    qkey = (wid, grp, mdt, seed_limit)
                    if qkey in qcache:
                        hist = qcache[qkey]
                    else:
                        try:
                            hist = dbu.get_recent_targets_by_group(
                                workface_id=wid,
                                spatiotemporal_group=grp,
                                measurement_date=mdt,
                                limit=int(seed_limit)
                            )
                        except Exception:
                            hist = None
                        qcache[qkey] = hist

                    if not hist or (not isinstance(hist, (list, tuple))):
                        continue

                    # 反转为时间正序
                    try:
                        hist_seq = list(reversed(hist))
                    except Exception:
                        hist_seq = list(hist)

                    for h in hist_seq:
                        if not isinstance(h, dict):
                            continue
                        seed_rows.append({
                            "workface_id": wid,
                            "spatiotemporal_group": grp,
                            "measurement_date": str(h.get("measurement_date", "")).strip() or None,
                            "drilling_cuttings_s": h.get("drilling_cuttings_s", None),
                            "gas_emission_velocity_q": h.get("gas_emission_velocity_q", None),
                            "gas_emission_q": h.get("gas_emission_q", None),
                            seed_flag_col: 1
                        })
                except Exception:
                    continue

            if not seed_rows:
                return df, 0

            # 拼接
            try:
                seed_df = pd.DataFrame(seed_rows)

                # 补 measurement_date_parsed（便于排序）
                try:
                    seed_df["measurement_date_parsed"] = pd.to_datetime(seed_df["measurement_date"], errors="coerce")
                except Exception:
                    pass

                df2 = pd.concat([seed_df, df], ignore_index=True, sort=False)

                if "measurement_date_parsed" not in df2.columns and "measurement_date" in df2.columns:
                    try:
                        df2["measurement_date_parsed"] = pd.to_datetime(df2["measurement_date"], errors="coerce")
                    except Exception:
                        pass

                # 排序（尽量）
                try:
                    if "workface_id" in df2.columns and "measurement_date_parsed" in df2.columns:
                        df2 = df2.sort_values(["workface_id", "measurement_date_parsed"], kind="mergesort")
                except Exception:
                    pass

                return df2, int(len(seed_df))
            except Exception:
                return df, 0

        except Exception:
            return df, 0
