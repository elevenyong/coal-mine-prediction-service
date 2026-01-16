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

        # 1) days_since_start / days_in_workface：必须计算（否则永远为0，时空交互特征失效）
        # 口径：
        # - 以“工作面维度”的 anchor_date 为起点：days = measurement_date - anchor_date
        # - anchor_date 优先从 DB 获取（若提供）；否则用本批次最早日期（不时间穿越）
        try:
            # 先确保 parsed 日期
            if "measurement_date_parsed" not in df.columns and "measurement_date" in df.columns:
                df["measurement_date_parsed"] = pd.to_datetime(df["measurement_date"], errors="coerce")

            if "measurement_date_parsed" in df.columns and "workface_id" in df.columns:
                # 选择一个可用的 db_utils（若存在）
                dbu = None
                for cand in [getattr(self, "_db_utils_for_enrich", None),
                             getattr(self, "_db_utils_for_train", None),
                             getattr(self, "_db_utils_for_predict", None)]:
                    if cand is not None:
                        dbu = cand
                        break

                # DB 获取 anchor 的候选 getter（不强绑某个名字）
                anchor_getter = None
                for name in ["get_workface_anchor_date", "get_anchor_date", "get_workface_start_date"]:
                    if dbu is not None and hasattr(dbu, name):
                        anchor_getter = getattr(dbu, name)
                        break

                # 逐 workface 计算 anchor_date，并生成 days_since_start/days_in_workface
                days_series = pd.Series(0.0, index=df.index)
                for wid, sub in df.groupby("workface_id", sort=False):
                    # anchor_date：优先 DB；否则批内最早日期
                    anchor_dt = None
                    if anchor_getter is not None:
                        try:
                            anchor_dt = anchor_getter(workface_id=int(float(wid)))
                        except TypeError:
                            try:
                                anchor_dt = anchor_getter(int(float(wid)))
                            except Exception:
                                anchor_dt = None
                        except Exception:
                            anchor_dt = None

                    try:
                        if anchor_dt is not None and str(anchor_dt).strip():
                            anchor_dt = pd.to_datetime(anchor_dt, errors="coerce")
                    except Exception:
                        anchor_dt = None

                    if anchor_dt is None or pd.isna(anchor_dt):
                        # 批内兜底：取该工作面本批最早日期（不时间穿越）
                        try:
                            anchor_dt = pd.to_datetime(sub["measurement_date_parsed"].min(), errors="coerce")
                        except Exception:
                            anchor_dt = None

                    if anchor_dt is None or pd.isna(anchor_dt):
                        # 仍拿不到 anchor，就保持 0，并留下明确告警（不掩盖问题）
                        logger.warning(
                            f"无法计算 days_since_start：workface_id={wid} 缺少可用 anchor_date 或 measurement_date")
                        continue

                    # days = (dt - anchor).days
                    try:
                        dt = sub["measurement_date_parsed"]
                        d = (dt - anchor_dt).dt.total_seconds() / 86400.0
                        d = d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        # 天数不应为负（出现负数通常是日期乱序或脏数据）
                        d = d.where(d >= 0, 0.0)
                        days_series.loc[sub.index] = d
                    except Exception:
                        continue

                df["days_since_start"] = days_series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                df["days_in_workface"] = df["days_since_start"]

            else:
                # 缺列则明确告警（而不是悄悄补0）
                miss = []
                if "workface_id" not in df.columns:
                    miss.append("workface_id")
                if "measurement_date_parsed" not in df.columns:
                    miss.append("measurement_date_parsed/measurement_date")
                if miss:
                    logger.warning(f"无法计算 days_since_start：缺少关键列 {miss}（将保持为0）")

            # 若仍不存在列，则补0（保持下游稳定）
            for c in ["days_since_start", "days_in_workface"]:
                if c not in df.columns:
                    df[c] = 0.0

        except Exception as e:
            logger.warning(f"计算 days_since_start/days_in_workface 失败（将保持为0）：{repr(e)}", exc_info=True)
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

        # 2. 工作面推进特征（advance_rate）
        # 说明（非常重要）：
        #   - distance_to_face 在你的业务里可能是“孔深/距工作面法向距离”（2m/3m/…/10m），
        #     也可能被历史口径当作“进尺/推进尺度”。两种口径不能混用。
        #   - 因此 advance_rate 的“推进尺度来源列”必须可配置：
        #       * 回采（work_stage=回采 或 roadway=mining）：默认用 face_advance_distance（或 distance_from_entrance）
        #         来计算推进速率；只有当你明确把 distance_to_face 定义为“进尺”时才选择 distance_to_face。
        #       * 掘进/其他：默认 face_advance_distance，其次 distance_from_entrance。
        # 关键点：
        #   - 先按“工作面+日期”聚合到天级（避免同一天150~300条样本导致 delta_days=0 产生inf/抖动）
        #   - 再把“天级推进速率”回填到当天的所有样本行
        df['advance_rate'] = 0.0  # 默认兜底
        if 'measurement_date_parsed' in df.columns and 'working_face' in df.columns:
            try:
                # 识别回采阶段（优先）
                is_mining = None
                try:
                    is_mining = (
                            (df.get('work_stage', '').astype(str) == '回采')
                            | (df.get('roadway', '').astype(str).str.lower() == 'mining')
                    )
                except Exception:
                    is_mining = None
                # 选择推进尺度来源列（可配置）
                def _pick_adv_source(is_mining_flag: bool):
                    if is_mining_flag:
                        src = self.config.get('SpatioTemporal', 'advance_rate_source_mining',
                                              fallback='face_advance_distance')
                    else:
                        src = self.config.get('SpatioTemporal', 'advance_rate_source_other',
                                              fallback='face_advance_distance')
                    src = str(src).strip()
                    alias = {
                        'entrance': 'distance_from_entrance',
                        'from_entrance': 'distance_from_entrance',
                        'advance': 'face_advance_distance',
                        'face': 'face_advance_distance',
                        'to_face': 'distance_to_face',
                        'depth': 'drilling_depth',
                    }
                    return alias.get(src, src)

                # ---------------------------
                # 2.1 回采：默认用 face_advance_distance / distance_from_entrance 计算推进速率
                # ---------------------------
                if is_mining is not None and bool(is_mining.any()):
                    adv_col = _pick_adv_source(True)
                    if adv_col not in df.columns:
                        for cand in ['face_advance_distance', 'distance_from_entrance', 'distance_to_face']:
                            if cand in df.columns:
                                adv_col = cand
                                break
                    if adv_col in df.columns:
                        tmp = df.loc[is_mining, ['working_face', 'measurement_date_parsed', adv_col]].copy()
                        tmp = tmp.rename(columns={adv_col: '__adv_base__'})
                        daily = (
                            tmp.groupby(['working_face', 'measurement_date_parsed'], as_index=False)['__adv_base__']
                            .max()
                        )
                        daily = daily.sort_values(['working_face', 'measurement_date_parsed'], kind='mergesort')
                        delta_days = daily.groupby('working_face')['measurement_date_parsed'].diff().dt.total_seconds() / 86400.0
                        try:
                            delta_days = delta_days.replace(0, np.nan)
                            delta_days = delta_days.where(delta_days > 0)
                        except Exception:
                            pass
                        delta_adv = daily.groupby('working_face')['__adv_base__'].diff()
                        rate = delta_adv / delta_days
                        try:
                            rate = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        except Exception:
                            rate = rate.fillna(0.0)
                        daily['advance_rate'] = rate
                        try:
                            if 'advance_rate' in df.columns:
                                df = df.drop(columns=['advance_rate'])
                        except Exception:
                            pass
                        df = df.merge(
                            daily[['working_face', 'measurement_date_parsed', 'advance_rate']],
                            on=['working_face', 'measurement_date_parsed'],
                            how='left'
                        )
                        try:
                            df['advance_rate'] = df['advance_rate'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        except Exception:
                            df['advance_rate'] = df['advance_rate'].fillna(0.0)

                # ---------------------------
                # 2.2 非回采：默认 face_advance_distance（若存在），否则 distance_from_entrance
                # ---------------------------
                else:
                    adv_col = _pick_adv_source(False)
                    if adv_col not in df.columns:
                        for cand in ['face_advance_distance', 'distance_from_entrance']:
                            if cand in df.columns:
                                adv_col = cand
                                break
                    if adv_col in df.columns:
                        tmp = df[['working_face', 'measurement_date_parsed', adv_col]].copy()
                        tmp = tmp.rename(columns={adv_col: '__adv_base__'})
                        daily = (
                            tmp.groupby(['working_face', 'measurement_date_parsed'], as_index=False)['__adv_base__']
                            .max()
                        )
                        daily = daily.sort_values(['working_face', 'measurement_date_parsed'], kind='mergesort')
                        delta_days = daily.groupby('working_face')['measurement_date_parsed'].diff().dt.total_seconds() / 86400.0
                        try:
                            delta_days = delta_days.replace(0, np.nan)
                            delta_days = delta_days.where(delta_days > 0)
                        except Exception:
                            pass
                        delta_adv = daily.groupby('working_face')['__adv_base__'].diff()
                        rate = delta_adv / delta_days
                        try:
                            rate = rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        except Exception:
                            rate = rate.fillna(0.0)
                        daily['advance_rate'] = rate
                        df = df.merge(
                            daily[['working_face', 'measurement_date_parsed', 'advance_rate']],
                            on=['working_face', 'measurement_date_parsed'],
                            how='left'
                        )
                        try:
                            df['advance_rate'] = df['advance_rate'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        except Exception:
                            df['advance_rate'] = df['advance_rate'].fillna(0.0)
            except Exception as _e:
                # 任何异常不影响主流程：推进速率统一回退为0
                try:
                    df['advance_rate'] = 0.0
                except Exception:
                    pass

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
                # ------------------- 新增：q 的“均值 + 滑动均值”融合增强（只增强 gas_emission_velocity_q） -------------------
                try:
                    q_col = "gas_emission_velocity_q"
                    if q_col in df.columns:
                        # 1) 计算“过去窗口”的滑动均值（只看历史：shift(1)）
                        #    window 默认 5（你可按数据密度调 5/7/9）
                        try:
                            q_roll_window = self.config.getint("SpatioTemporal", "q_roll_window", fallback=5)
                            if q_roll_window <= 1:
                                q_roll_window = 5
                        except Exception:
                            q_roll_window = 5

                        q_shift = df.groupby(group_key)[q_col].shift(1)

                        q_roll_mean = (
                            q_shift.groupby(df[group_key])
                            .rolling(window=q_roll_window, min_periods=1)
                            .mean()
                            .reset_index(level=0, drop=True)
                        )
                        q_roll_mean = q_roll_mean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        df[f"{q_col}_rolling_mean"] = q_roll_mean

                        # 2) 融合：历史 expanding mean + rolling mean
                        #    默认 rolling 权重大一点（更抗噪），但仍保留长期均值
                        try:
                            w_roll = self.config.getfloat("SpatioTemporal", "q_roll_weight", fallback=0.65)
                        except Exception:
                            w_roll = 0.65
                        if w_roll < 0:
                            w_roll = 0.0
                        if w_roll > 1:
                            w_roll = 1.0
                        w_hist = 1.0 - w_roll

                        hist_name = f"{q_col}_historical_mean"
                        if hist_name not in df.columns:
                            df[hist_name] = 0.0

                        df[f"{q_col}_hist_fused"] = (
                                w_hist * df[hist_name].astype(float) + w_roll * df[f"{q_col}_rolling_mean"].astype(
                            float)
                        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                except Exception as _e:
                    logger.warning(f"q 历史统计融合增强失败（已忽略）：{repr(_e)}", exc_info=True)
                # ------------------- 新增结束 -------------------

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

    # =====================================================================
    # [唯一生效版本] 预测阶段增强特征回填函数
    # 严格约束：
    #   1. 缺 measurement_date 直接跳过（避免时间穿越）
    #   2. 不允许隐式调用 get_latest_feature_cache 的“取最新”语义
    #   3. 本文件中禁止再出现同名函数定义，否则属于严重逻辑错误
    # =====================================================================
    def _fill_enhanced_from_feature_cache(self, df, db_utils):
        """
        预测阶段：从 t_feature_cache 回填增强特征（historical_mean/trend/advance_rate等）
        db_utils 需要实现类似方法：get_feature_cache_by_keys(.) 或等价接口
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

        # 需要的键：workface_id + spatiotemporal_group
        key_cols = ["workface_id", "spatiotemporal_group"]
        if not all(c in df.columns for c in key_cols):
            return df

        # --- 新增：告警只打一次，避免逐行刷屏 ---
        warned_no_date = False

        # 逐行回填（数据量不大时可接受；若后续需要可再批量优化）
        try:
            for idx, row in df.iterrows():
                try:
                    wid = int(float(row.get("workface_id")))
                    grp = str(row.get("spatiotemporal_group", "")).strip()
                    if not grp:
                        continue

                    # P0：预测阶段必须携带 measurement_date（否则会时间穿越/或退化为全0）
                    mdt = None
                    if "measurement_date" in df.columns:
                        mdt = str(row.get("measurement_date", "")).strip() or None

                    if not mdt:
                        # 记录一次性告警（不刷屏），并跳过回填（不掩盖问题）
                        try:
                            if not hasattr(self, "_enhance_backfill_warned_no_date"):
                                self._enhance_backfill_warned_no_date = False
                            if not self._enhance_backfill_warned_no_date:
                                logger.warning(
                                    "预测阶段缺少 measurement_date：将跳过增强特征回填以避免时间穿越。"
                                    "请在预测输入中补齐 measurement_date（YYYY-MM-DD）。"
                                )
                                self._enhance_backfill_warned_no_date = True
                        except Exception:
                            pass
                        # 统计信息（供上层返回 partial_success 用）
                        try:
                            if not hasattr(self, "_enhance_backfill_stats"):
                                self._enhance_backfill_stats = {"missing_measurement_date": 0, "cache_miss": 0}
                            self._enhance_backfill_stats["missing_measurement_date"] += 1
                        except Exception:
                            pass
                        continue

                    # 有日期：只能用“按日期之前最近一条”，禁止退化为取最新
                    try:
                        cached = getter(workface_id=wid, spatiotemporal_group=grp, measurement_date=mdt)
                    except TypeError:
                        # 兼容不同签名
                        cached = getter(wid, grp, mdt)

                    if not cached or not isinstance(cached, dict):
                        try:
                            if not hasattr(self, "_enhance_backfill_stats"):
                                self._enhance_backfill_stats = {"missing_measurement_date": 0, "cache_miss": 0}
                            self._enhance_backfill_stats["cache_miss"] += 1
                        except Exception:
                            pass
                        continue

                    for k, v in cached.items():
                        # 只回填增强列，不覆盖基础输入列
                        if k in df.columns and (
                                k.endswith("_historical_mean")
                                or k.endswith("_trend")
                                or k.endswith("_rolling_mean")
                                or k.endswith("_hist_fused")
                                or k in ["advance_rate"]
                        ):
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

    # =====================================================================
    # [唯一生效版本] 训练阶段 bridge 历史特征接续函数
    # 严格约束：
    #   1. 优先使用 measurement_date 调用 DB（避免时间穿越）
    #   2. 仅修补“组内首条样本”的历史/趋势退化问题
    #   3. 本文件中禁止再出现同名函数定义，否则属于严重逻辑错误
    # =====================================================================
    def _bridge_training_history_from_feature_cache(self, df, db_utils, group_key):
        """
        训练阶段：bridge 接续（仅修补“组内首条记录”的历史/趋势退化问题）
        db_utils 需要实现类似方法：get_latest_enhanced_by_group(...) 或等价接口
        """
        if df is None or len(df) == 0 or db_utils is None:
            return df
        if group_key not in df.columns:
            return df
        # 优先使用“按日期之前最近一条”的接口（避免时间穿越）
        getter = None
        for cand in ["get_latest_feature_cache_by_group", "get_latest_feature_cache", "get_latest_enhanced_by_group",
                     "get_feature_cache"]:
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
                    # P0：训练 bridge 必须传 measurement_date（避免时间穿越/避免退化）
                    mdt = None
                    if "measurement_date" in df.columns:
                        try:
                            v = df.at[first_idx, "measurement_date"]
                            mdt = str(v).strip() if v is not None else None
                        except Exception:
                            mdt = None
                    if not mdt and "measurement_date_parsed" in df.columns:
                        try:
                            mdt = pd.to_datetime(df.at[first_idx, "measurement_date_parsed"], errors="coerce").strftime(
                                "%Y-%m-%d")
                        except Exception:
                            mdt = None

                    if not mdt:
                        logger.warning(
                            f"训练bridge缺少 measurement_date（wid={wid}, grp={grp}）：跳过历史接续（需补齐日期链路）"
                        )
                        try:
                            if not hasattr(self, "_enhance_backfill_stats"):
                                self._enhance_backfill_stats = {"missing_measurement_date": 0, "cache_miss": 0}
                            self._enhance_backfill_stats["missing_measurement_date"] += 1
                        except Exception:
                            pass
                        continue

                    try:
                        cached = getter(workface_id=wid, spatiotemporal_group=grp, measurement_date=mdt)
                    except TypeError:
                        cached = getter(wid, grp, mdt)

                    if not cached or not isinstance(cached, dict):
                        try:
                            if not hasattr(self, "_enhance_backfill_stats"):
                                self._enhance_backfill_stats = {"missing_measurement_date": 0, "cache_miss": 0}
                            self._enhance_backfill_stats["cache_miss"] += 1
                        except Exception:
                            pass
                        continue

                    # 仅在当前为0时补上上一条增强值
                    for k, v in cached.items():
                        if k in df.columns and (
                                k.endswith("_historical_mean")
                                or k.endswith("_trend")
                                or k.endswith("_rolling_mean")
                                or k.endswith("_hist_fused")
                                or k in ["advance_rate"]
                        ):
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
            # 关键修复：
            # - 预测输入来自 JSON 时，很多“数字列”可能是 object/string（例如 "9.0"）
            # - sklearn 的数值管道会对 object 调 np.isnan -> 直接报 ufunc isnan 不支持
            # 因此：优先按配置的 base_numeric 强制转数值，再做 inf/NaN 清洗
            numeric_candidates = []
            try:
                if hasattr(self, "base_numeric") and isinstance(self.base_numeric, list):
                    numeric_candidates = [c for c in self.base_numeric if c in df.columns]
            except Exception:
                numeric_candidates = []

            # 对配置的数值列做强制数值化（即使 dtype 仍是 object 也会被转）
            if numeric_candidates:
                for c in numeric_candidates:
                    try:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    except Exception:
                        pass

            # 再补一层：对当前已识别为数值 dtype 的列统一做 inf/NaN 清洗
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
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

            # spatiotemporal_group 为强一致路径；coord_hash 为回退路径（去掉distance bucket影响）
            need_cols = ["workface_id", "spatiotemporal_group", "measurement_date"]
            for c in need_cols:
                if c not in df.columns:
                    return df, 0

            if seed_limit is None:
                try:
                    seed_limit = self.config.getint("SpatioTemporal", "history_seed_limit", fallback=20)
                except Exception:
                    seed_limit = 20

            try:
                seed_limit = int(seed_limit)
                if seed_limit <= 0:
                    return df, 0
            except Exception:
                seed_limit = 20

            # lookback_days：仅回捞最近N天（避免远古数据污染趋势）
            try:
                lookback_days = self.config.getint("SpatioTemporal", "history_seed_lookback_days", fallback=30)
            except Exception:
                lookback_days = 30
            try:
                lookback_days = int(lookback_days)
                if lookback_days <= 0:
                    lookback_days = None
            except Exception:
                lookback_days = 30

            # 去重聚合：按 measurement_date 聚合（同一天多条均值）
            try:
                dedupe_by_date = self.config.getboolean("SpatioTemporal", "history_seed_dedupe_by_date", fallback=True)
            except Exception:
                dedupe_by_date = True

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
                    # 额外取当前组在 df 中的一个 coord_hash（用于回退1）
                    # 注意：min_dates里没有coord_hash，因此要从原df里找
                    ch = None
                    try:
                        if "coord_hash" in df.columns:
                            # 同 wid+grp 的任意一条即可
                            tmp = df[(df["workface_id"] == wid_raw) & (df["spatiotemporal_group"] == grp_raw)]
                            if tmp is not None and len(tmp) > 0:
                                ch = str(tmp.iloc[0].get("coord_hash", "")).strip() or None
                    except Exception:
                        ch = None
                    qkey = (wid, grp, mdt, seed_limit, ch)
                    if qkey in qcache:
                        hist = qcache[qkey]
                    else:
                        hist = None
                        # 1) 强一致：按 group
                        try:
                            # 新版：支持 lookback_days + dedupe_by_date（旧版db_utils也能兼容，TypeError会回退）
                            try:
                                hist = dbu.get_recent_targets_by_group(
                                    workface_id=wid,
                                    spatiotemporal_group=grp,
                                    measurement_date=mdt,
                                    limit=int(seed_limit),
                                    lookback_days=lookback_days,
                                    dedupe_by_date=dedupe_by_date
                                )
                            except TypeError:
                                hist = dbu.get_recent_targets_by_group(
                                    workface_id=wid,
                                    spatiotemporal_group=grp,
                                    measurement_date=mdt,
                                    limit=int(seed_limit)
                                )
                        except Exception:
                            hist = None

                        # 2) 回退1：按 coord_hash（去掉bucket）
                        if (not hist) and ch and hasattr(dbu, "get_recent_targets_by_coord_hash"):
                            try:
                                hist = dbu.get_recent_targets_by_coord_hash(
                                    workface_id=wid,
                                    coord_hash=ch,
                                    measurement_date=mdt,
                                    limit=int(seed_limit)
                                )
                            except Exception:
                                hist = None

                        # 3) 回退2：按 workface（兜底保证有seed）
                        if (not hist) and hasattr(dbu, "get_recent_targets_by_workface"):
                            try:
                                hist = dbu.get_recent_targets_by_workface(
                                    workface_id=wid,
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

                    # 二次兜底：按 measurement_date 聚合去重（均值），避免同一天多条导致seed爆量
                    if dedupe_by_date:
                        try:
                            acc = {}
                            for h in hist_seq:
                                if not isinstance(h, dict):
                                    continue
                                dkey = str(h.get("measurement_date", "")).strip()
                                if not dkey:
                                    continue
                                s = h.get("drilling_cuttings_s", None)
                                qv = h.get("gas_emission_velocity_q", None)
                                gq = h.get("gas_emission_q", None)

                                if dkey not in acc:
                                    acc[dkey] = {"n": 0, "s": 0.0, "q": 0.0, "gq": 0.0,
                                                 "has_s": 0, "has_q": 0, "has_gq": 0}
                                a = acc[dkey]
                                a["n"] += 1
                                if s is not None and str(s) != "nan":
                                    try:
                                        a["s"] += float(s);
                                        a["has_s"] += 1
                                    except Exception:
                                        pass
                                if qv is not None and str(qv) != "nan":
                                    try:
                                        a["q"] += float(qv);
                                        a["has_q"] += 1
                                    except Exception:
                                        pass
                                if gq is not None and str(gq) != "nan":
                                    try:
                                        a["gq"] += float(gq);
                                        a["has_gq"] += 1
                                    except Exception:
                                        pass

                            # 保持时间正序
                            hist_seq2 = []
                            for dkey in sorted(acc.keys()):
                                a = acc[dkey]
                                hist_seq2.append({
                                    "measurement_date": dkey,
                                    "drilling_cuttings_s": (a["s"] / a["has_s"]) if a["has_s"] > 0 else None,
                                    "gas_emission_velocity_q": (a["q"] / a["has_q"]) if a["has_q"] > 0 else None,
                                    "gas_emission_q": (a["gq"] / a["has_gq"]) if a["has_gq"] > 0 else None,
                                })
                            hist_seq = hist_seq2
                        except Exception:
                            pass

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

