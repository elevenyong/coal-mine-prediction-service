"""
煤矿瓦斯风险预测系统 - 模型管理模块
包含：模型保存/加载、版本管理、数据库交互
"""
import os
import shutil
import joblib
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

            # 清理旧备份（仅保留最近5份）
            backups = sorted(os.listdir(backup_root))
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    old_backup_dir = os.path.join(backup_root, old_backup)
                    shutil.rmtree(old_backup_dir, ignore_errors=True)
                    logger.info(f"清理旧备份：{old_backup_dir}")

        except Exception as e:
            logger.error(f"保存模型失败：{str(e)}", exc_info=True)

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
            with db_utils._get_connection() as conn:
                query_sql = text("SELECT COUNT(*) AS total FROM t_prediction_parameters")
                result = conn.execute(query_sql)
                row = result.fetchone()
                total = row[0] if row else 0
                if total < 0:
                    raise ValueError(f"样本数异常：{total}（必须≥0）")
                logger.info(f"从数据库查询累计样本数：{total}")
                return total
        except Exception as e:
            logger.warning(f"查询累计样本数失败：{str(e)}")
            return 0

    def get_recent_data_from_db(self, db_utils, limit=None):
        """
        从数据库读取最近N条数据

        :param db_utils: 数据库工具实例
        :param limit: 读取样本数限制
        :return: pandas.DataFrame，读取的数据
        """
        try:
            db_conf = db_utils.db_config
            db_url = (
                f"mysql+pymysql://{db_conf['user']}:{db_conf['password']}@"
                f"{db_conf['host']}:{db_conf['port']}/{db_conf['db']}?charset={db_conf['charset']}"
            )
            engine = create_engine(db_url)

            if limit:
                query_sql = text(f"""
                    SELECT * FROM t_prediction_parameters 
                    ORDER BY distance_from_entrance DESC 
                    LIMIT :limit
                """)
                params = {"limit": limit}
            else:
                query_sql = text("""
                    SELECT * FROM t_prediction_parameters 
                    ORDER BY distance_from_entrance DESC
                """)
                params = {}

            with engine.connect() as conn:
                df = pd.read_sql(query_sql, conn, params=params)

            if not df.empty and 'distance_from_entrance' in df.columns:
                df = df.sort_values(by='distance_from_entrance', ascending=True).reset_index(drop=True)
                logger.info(f"数据已按掘进距离升序排序（样本数：{len(df)}）")

            logger.info(f"数据库读取完成，有效样本数：{len(df)}")
            return df
        except Exception as e:
            logger.error(f"从数据库读取数据失败：{str(e)}", exc_info=True)
            raise

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
