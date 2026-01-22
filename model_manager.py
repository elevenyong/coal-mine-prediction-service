"""
煤矿瓦斯风险预测系统 - 模型管理模块
包含：模型保存/加载、版本管理、数据库交互
"""
import os
import re
import shutil
import joblib
import json
from loguru import logger
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text


class ModelManager:
    """模型管理器"""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
        self.features_path = os.path.join(self.model_dir, "training_features.pkl")
        self.algorithm_path = os.path.join(self.model_dir, "algorithm.pkl")
        self.lock_file_path = os.path.join(self.model_dir, "model.lock")
        self.meta_path = os.path.join(self.model_dir, "model_meta.json")

    def save_model(self, models, preprocessor, training_features, target_features):
        """
        保存模型组件（预处理模型、特征列表、模型）并创建备份

        :param models: 模型字典
        :param preprocessor: 特征预处理器
        :param training_features: 训练特征列表
        :param target_features: 目标特征列表
        """
        try:
            # 保存预处理模型
            if preprocessor:
                joblib.dump(preprocessor, self.preprocessor_path)
                logger.debug(f"已保存预处理模型至：{self.preprocessor_path}")

            # 保存训练特征列表
            if training_features:
                joblib.dump(training_features, self.features_path)
                logger.debug(f"已保存训练特征列表至：{self.features_path}")

            # 检测算法类型并保存
            algorithm_type = self._detect_algorithm_type(models)
            joblib.dump(algorithm_type, self.algorithm_path)
            logger.debug(f"检测到算法类型：{algorithm_type}")

            # 保存每个目标的模型
            for target, model in models.items():
                if algorithm_type == "lightgbm":
                    model_path = os.path.join(self.model_dir, f"model_{target}.txt")
                    if model is not None:
                        model.save_model(model_path)
                        logger.debug(f"已保存目标[{target}]的LightGBM模型至：{model_path}")
                elif algorithm_type == "xgboost":
                    model_path = os.path.join(self.model_dir, f"model_{target}.json")
                    if model is not None:
                        model.save_model(model_path)
                        logger.debug(f"已保存目标[{target}]的XGBoost模型至：{model_path}")
                else:
                    raise ValueError(f"不支持的算法类型: {algorithm_type}")

            # 创建时间戳备份
            backup_root = os.path.join(self.model_dir, "backup")
            os.makedirs(backup_root, exist_ok=True)
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(backup_root, backup_timestamp)
            os.makedirs(backup_dir, exist_ok=True)

            # 复制核心文件到备份目录
            for file in [self.preprocessor_path, self.features_path, self.algorithm_path]:
                if os.path.exists(file):
                    shutil.copy(file, backup_dir)
            for target in models.keys():
                if algorithm_type == "lightgbm":
                    model_file = f"model_{target}.txt"
                else:
                    model_file = f"model_{target}.json"
                model_path = os.path.join(self.model_dir, model_file)
                if os.path.exists(model_path):
                    shutil.copy(model_path, backup_dir)

            logger.info(f"模型备份完成，备份目录：{backup_dir}，算法类型：{algorithm_type}")
            # 写入线上模型元数据（供预测接口回传使用的模型版本/算法等）
            try:
                # 关键：不要覆盖已有 baseline_rmse（否则重启后又变 null）
                old_meta = self.get_active_model_info() or {}
                meta = {
                    "active_version": backup_timestamp,
                    "algorithm": algorithm_type,
                    "targets": list((models or {}).keys()),
                    "saved_at": datetime.now().isoformat(timespec="seconds")
                }
                # 继承旧的 baseline 信息（如果已有）
                if isinstance(old_meta, dict):
                    if old_meta.get("baseline_rmse") is not None:
                        meta["baseline_rmse"] = old_meta.get("baseline_rmse")
                    if old_meta.get("baseline_eval") is not None:
                        meta["baseline_eval"] = old_meta.get("baseline_eval")
                self._write_active_meta(meta, backup_dir=backup_dir)
            except Exception:
                pass

            # 清理旧备份（仅保留最近5份）
            backups = sorted(os.listdir(backup_root))
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    old_backup_dir = os.path.join(backup_root, old_backup)
                    shutil.rmtree(old_backup_dir, ignore_errors=True)
                    logger.info(f"清理旧备份：{old_backup_dir}")

        except Exception as e:
            logger.error(f"保存模型失败：{str(e)}", exc_info=True)

    def _write_active_meta(self, meta: dict, backup_dir: str = None):
        """写入线上模型元数据（用于预测接口回传模型版本/精度等信息）。"""
        try:
            if not isinstance(meta, dict):
                meta = {}
            meta.setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            if backup_dir:
                try:
                    os.makedirs(backup_dir, exist_ok=True)
                    shutil.copy(self.meta_path, os.path.join(backup_dir, "model_meta.json"))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"写入model_meta.json失败：{e}")

    def get_active_model_info(self) -> dict:
        """获取当前线上模型信息（版本/算法/保存时间等）。"""
        info = {}
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    info = json.load(f) or {}
        except Exception as e:
            logger.debug(f"读取model_meta.json失败：{e}")

        # 兜底：补齐 algorithm / active_version
        if not info.get("algorithm"):
            try:
                if os.path.exists(self.algorithm_path):
                    info["algorithm"] = joblib.load(self.algorithm_path)
            except Exception:
                pass

        if not info.get("active_version"):
            try:
                backup_root = os.path.join(self.model_dir, "backup")
                if os.path.exists(backup_root):
                    backups = sorted(os.listdir(backup_root), reverse=True)
                    if backups:
                        info["active_version"] = backups[0]
            except Exception:
                pass

        return info or {}

    def save_candidate_model(self, models, preprocessor, training_features, target_features, tag: str = ""):
        """
        保存“候选模型”（不覆盖线上模型）。用于方案A：评估未通过时保留训练产物以便排查/复现。
        - 输出目录：<model_dir>/candidates/<timestamp>[_<tag>]/
        """
        try:
            cand_root = os.path.join(self.model_dir, "candidates")
            os.makedirs(cand_root, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_tag = ""
            # Windows + LightGBM: 候选目录名强制ASCII，避免中文/Unicode导致 save_model 写文件失败
            def _to_ascii_safe(s: str, max_len: int = 40) -> str:
                if not s:
                    return ""
                s = str(s)
                # 仅保留 ASCII 可见安全字符：A-Z a-z 0-9 _ - .
                # 其余（包含中文等Unicode）全部替换为 "_"
                s2 = []
                for ch in s:
                    o = ord(ch)
                    if (48 <= o <= 57) or (65 <= o <= 90) or (97 <= o <= 122) or (ch in "_-."):
                        s2.append(ch)
                    else:
                        s2.append("_")
                out = "".join(s2)
                out = re.sub(r"_+", "_", out).strip("_")
                return out[:max_len]

            safe_tag = _to_ascii_safe(tag, max_len=40)
            suffix = f"_{safe_tag}" if safe_tag else ""
            out_dir = os.path.join(cand_root, f"{ts}{suffix}")
            os.makedirs(out_dir, exist_ok=True)
            self._save_model_to_dir(models, preprocessor, training_features, target_features, out_dir)
            logger.info(f"候选模型已保存：{out_dir}")
            return out_dir
        except Exception as e:
            logger.error(f"保存候选模型失败：{str(e)}", exc_info=True)
            raise

    def _save_model_to_dir(self, models, preprocessor, training_features, target_features, out_dir: str):
        """内部：将模型组件保存到指定目录（不影响线上路径）。"""
        if preprocessor:
            joblib.dump(preprocessor, os.path.join(out_dir, "preprocessor.pkl"))
        if training_features:
            joblib.dump(training_features, os.path.join(out_dir, "training_features.pkl"))

        algorithm_type = self._detect_algorithm_type(models)
        joblib.dump(algorithm_type, os.path.join(out_dir, "algorithm_type.pkl"))

        if target_features:
            try:
                joblib.dump(target_features, os.path.join(out_dir, "target_features.pkl"))
            except Exception:
                pass

        for target, model in (models or {}).items():
            if algorithm_type == "lightgbm":
                model_path = os.path.join(out_dir, f"model_{target}.txt")
                if model is not None:
                    # LightGBM(C++) 在部分Windows环境对反斜杠路径不稳定，统一转换为正斜杠更稳
                    model.save_model(model_path.replace("\\", "/"))
            elif algorithm_type == "xgboost":
                model_path = os.path.join(out_dir, f"model_{target}.json")
                if model is not None:
                    model.save_model(model_path)
            else:
                raise ValueError(f"不支持的算法类型: {algorithm_type}")

    def _detect_algorithm_type(self, models):
        """检测模型使用的算法类型"""
        if not models:
            return "unknown"

        # 检查第一个模型来判断算法类型
        first_model = next(iter(models.values()))

        # 更精确的算法类型检测
        if hasattr(first_model, 'save_model') and hasattr(first_model, 'num_trees'):
            # LightGBM 模型
            return "lightgbm"
        elif hasattr(first_model, 'save_model') and hasattr(first_model, 'get_dump'):
            # XGBoost 模型
            return "xgboost"
        else:
            # 通过类名判断
            class_name = type(first_model).__name__.lower()
            if 'lgb' in class_name or 'lightgbm' in class_name:
                return "lightgbm"
            elif 'xgb' in class_name or 'xgboost' in class_name:
                return "xgboost"
            else:
                logger.warning(f"无法识别的模型类型: {class_name}")
                return "unknown"

    def load_model(self, target_features):
        """
        加载已有模型组件

        :param target_features: 目标特征列表
        :return: tuple (preprocessor, training_features, models, is_trained)
        """
        import lightgbm as lgb
        import xgboost as xgb

        try:
            # 检查核心文件是否存在
            core_files_exist = (os.path.exists(self.preprocessor_path) and
                                os.path.exists(self.features_path) and
                                os.path.exists(self.algorithm_path))
            if not core_files_exist:
                return None, None, {}, False

            # 加载算法类型
            algorithm_type = joblib.load(self.algorithm_path)
            logger.debug(f"加载模型算法类型：{algorithm_type}")

            # 加载预处理模型与特征列表
            preprocessor = joblib.load(self.preprocessor_path)
            training_features = joblib.load(self.features_path)
            logger.debug(f"加载预处理模型成功，训练特征数：{len(training_features)}")

            # 加载每个目标的模型
            models = {}
            loaded_targets = []
            for target in target_features:
                if algorithm_type == "lightgbm":
                    model_path = os.path.join(self.model_dir, f"model_{target}.txt")
                elif algorithm_type == "xgboost":
                    model_path = os.path.join(self.model_dir, f"model_{target}.json")
                else:
                    logger.error(f"不支持的算法类型: {algorithm_type}")
                    continue

                if os.path.exists(model_path):
                    try:
                        if algorithm_type == "lightgbm":
                            model = lgb.Booster(model_file=model_path)
                            tree_count = model.num_trees()
                        elif algorithm_type == "xgboost":
                            model = xgb.Booster()
                            model.load_model(model_path)
                            tree_count = len(model.get_dump())

                        logger.debug(f"加载目标[{target}]的{algorithm_type}模型成功，树数量：{tree_count}")

                        if tree_count <= 1:
                            logger.warning(f"目标[{target}]的模型树数量异常（{tree_count}），可能存在问题")

                        models[target] = model
                        loaded_targets.append(target)
                    except Exception as e:
                        logger.error(f"加载目标[{target}]的模型失败：{str(e)}")
                        continue

            # 更新训练状态
            is_trained = len(loaded_targets) == len(target_features)
            if is_trained:
                logger.info(f"模型加载成功：{len(loaded_targets)}个目标{algorithm_type}模型+预处理模型")
            else:
                logger.warning(
                    f"模型加载不完整：仅加载{len(loaded_targets)}/{len(target_features)}个模型"
                )

            return preprocessor, training_features, models, is_trained

        except Exception as e:
            logger.error(f"加载模型失败：{str(e)}", exc_info=True)
            return None, None, {}, False

    def rollback_model(self, backup_index, target_features):
        """
        模型回滚到指定备份

        :param backup_index: 备份索引
        :param target_features: 目标特征列表
        :return: dict，回滚结果
        """
        try:
            backup_root = os.path.join(self.model_dir, "backup")
            if not os.path.exists(backup_root):
                msg = "备份目录不存在，无法回滚"
                logger.warning(msg)
                return {"success": False, "message": msg}

            # 获取备份列表
            backups = sorted(os.listdir(backup_root), reverse=True)
            if not backups:
                msg = "无任何模型备份，无法回滚"
                logger.warning(msg)
                return {"success": False, "message": msg}

            # 验证备份索引
            if abs(backup_index) > len(backups):
                msg = f"备份索引 {backup_index} 超出范围，共 {len(backups)} 个备份"
                logger.warning(msg)
                return {"success": False, "message": msg}

            # 恢复指定备份
            target_backup = backups[backup_index]
            target_dir = os.path.join(backup_root, target_backup)
            logger.info(f"回滚到备份：{target_backup}")

            # 恢复核心文件
            for file in ["preprocessor.pkl", "training_features.pkl", "algorithm.pkl"]:
                src = os.path.join(target_dir, file)
                dst = os.path.join(self.model_dir, file)
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    logger.info(f"恢复文件：{file}")

            # 恢复每个目标的模型文件
            algorithm_path = os.path.join(target_dir, "algorithm.pkl")
            if os.path.exists(algorithm_path):
                algorithm_type = joblib.load(algorithm_path)
                for target in target_features:
                    if algorithm_type == "lightgbm":
                        model_file = f"model_{target}.txt"
                    elif algorithm_type == "xgboost":
                        model_file = f"model_{target}.json"
                    else:
                        logger.error(f"不支持的算法类型: {algorithm_type}")
                        continue

                    src = os.path.join(target_dir, model_file)
                    dst = os.path.join(self.model_dir, model_file)
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                        logger.info(f"恢复模型文件：{model_file}")
            # 回滚后更新线上模型元数据（active_version 指向目标备份）
            try:
                meta = self.get_active_model_info()
                meta.update({
                    "active_version": target_backup,
                    "restored_from_backup": True,
                    "restored_at": datetime.now().isoformat(timespec="seconds")
                })
                self._write_active_meta(meta, backup_dir=None)
            except Exception:
                pass

            return {
                "success": True,
                "message": f"模型回滚到备份：{target_backup}",
                "backup_timestamp": target_backup
            }
        except Exception as e:
            logger.error(f"回滚失败：{str(e)}", exc_info=True)
            return {"success": False, "message": str(e)}

    def get_total_samples_from_db(self, db_utils):
        """
        从数据库查询累计样本数

        :param db_utils: 数据库工具实例
        :return: int，累计样本数
        """
        try:
            # 统一用 SQLAlchemy engine.connect()（MySQL/SQLite 通用且更稳定）
            with db_utils.engine.connect() as conn:
                query_sql = text("SELECT COUNT(*) AS total FROM t_prediction_parameters")
                result = conn.execute(query_sql)
                row = result.fetchone()
                total = int((row[0] if row else 0) or 0)  # None安全：COUNT(*) 理论不为None，但做兜底更稳
                if total < 0:
                    raise ValueError(f"样本数异常：{total}（必须≥0）")
                logger.info(f"从数据库查询累计样本数：{total}")
                return total
        except Exception as e:
            logger.warning(f"查询累计样本数失败：{str(e)}")
            return 0

    def get_recent_data_from_db(self, db_utils, limit=None):
        """
        从数据库读取最近N条数据（改为统一列清单生成器，避免 SELECT *）
        - pred 表字段：来自 build_join_column_lists 的 p_cols
        - feature_cache 增强字段：来自 build_join_column_lists 的 f_cols（统一 AS，避免重复列名）
        - 仍按 distance_from_entrance 读取最新，再按升序返回（与你原逻辑一致）
        """
        try:
            if db_utils is None:
                raise ValueError("db_utils 不能为空")

            # LIMIT 兼容（避免绑定问题，直接拼接字面量）
            limit_sql = ""
            if limit is not None:
                try:
                    lim = int(limit)
                    if lim > 0:
                        limit_sql = f"LIMIT {lim}"
                except Exception:
                    limit_sql = ""

            # 统一列清单生成器：p/f 两表列名（禁止 gas_emission_q / advance_rate 从 pred 表读取等）
            if not hasattr(db_utils, "build_join_column_lists"):
                raise ValueError("db_utils 未实现 build_join_column_lists，无法替换 SELECT *")

            p_cols, f_cols = db_utils.build_join_column_lists(
                needed_cols=None,
                include_targets=True
            )

            # 组装 SELECT（feature 列统一 AS 成原名）
            select_cols = [f"p.{c}" for c in p_cols] + [f"f.{c} AS {c}" for c in f_cols]
            if not select_cols:
                return pd.DataFrame()

            sql = text(f"""
                SELECT
                    {", ".join(select_cols)}
                FROM t_prediction_parameters p
                LEFT JOIN t_feature_cache f
                  ON p.id = f.pred_id
                ORDER BY p.distance_from_entrance DESC, p.id DESC
                {limit_sql}
            """)

            with db_utils.engine.connect() as conn:
                df = pd.read_sql(sql, conn)

            # 保持你原来的返回口径：升序
            if df is not None and (not df.empty) and "distance_from_entrance" in df.columns:
                df = df.sort_values(by="distance_from_entrance", ascending=True).reset_index(drop=True)

            logger.info(f"数据库读取完成，有效样本数：{len(df) if df is not None else 0}")
            # P0：measurement_date 是时序增强与评估可信性的关键字段，缺失会导致特征回填退化
            try:
                if "measurement_date" not in df.columns:
                    logger.warning("读取样本缺少 measurement_date 列：时序增强/feature_cache 回填将退化，评估可能被标记为 evaluation_invalid")
                else:
                    miss_ratio = float(df["measurement_date"].isna().mean())
                    if miss_ratio > 0.01:
                        logger.warning(f"读取样本 measurement_date 缺失率={miss_ratio:.2%}：将导致部分样本无法回填增强特征")
            except Exception:
                pass
            return df if df is not None else pd.DataFrame()

        except Exception as e:
            logger.error(f"从数据库读取数据失败：{str(e)}", exc_info=True)
            return pd.DataFrame()

    def get_locked_algorithm(self):
        """
        获取已保存模型对应的算法类型（用于锁定算法）
        :return: "lightgbm" / "xgboost" / None
        """
        try:
            if os.path.exists(self.algorithm_path):
                algorithm_type = joblib.load(self.algorithm_path)
                if algorithm_type in ("lightgbm", "xgboost"):
                    return algorithm_type
        except Exception as e:
            logger.warning(f"读取algorithm.pkl失败：{e}")
        return None
