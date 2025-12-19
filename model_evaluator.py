"""
煤矿瓦斯风险预测系统 - 模型评估模块
包含：模型性能评估、性能监控、自动回滚
依赖：scikit-learn、numpy
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
from sqlalchemy import create_engine, text

from config_utils import ConfigUtils


class ModelEvaluator(ConfigUtils):
    """模型评估器"""

    def __init__(self, config_path="config.ini"):
        super().__init__(config_path)
        self.eval_size = self.config.getint("ModelEval", "eval_size", fallback=200)
        self.perf_drop_ratio = self.config.getfloat("ModelEval", "perf_drop_ratio", fallback=0.10)
        self.perf_window = self.config.getint("ModelEval", "perf_window", fallback=2)
        self.small_data_threshold = self.config.getint("ModelEval", "small_data_threshold", fallback=10)
        self.medium_data_threshold = self.config.getint("ModelEval", "medium_data_threshold", fallback=20)
        agg_weights_str = self.config.get("ModelEval", "agg_weights", fallback="1,1,1,1")
        try:
            self.agg_weights = [float(x) for x in agg_weights_str.split(",")]
        except Exception:
            self.agg_weights = [1.0, 1.0, 1.0, 1.0]
            logger.warning(f"聚合权重解析失败（配置：{agg_weights_str}），使用默认值[1.0,1.0,1.0,1.0]")
    def evaluate_model(self, models, preprocessor, training_features, target_features,
                       fitted_feature_order, db_utils, eval_size=None, eval_df=None):
        """
        公开方法：模型评估（计算RMSE/MAE/R²，支持外部传入评估数据）

        :param models: 模型字典
        :param preprocessor: 特征预处理器
        :param training_features: 训练特征列表
        :param target_features: 目标特征列表
        :param fitted_feature_order: 训练时的特征顺序
        :param db_utils: 数据库工具实例
        :param eval_size: 评估样本数
        :param eval_df: 外部评估数据
        :return: dict，评估结果
        """
        self._print_header("模型性能评估")
        try:
            if not models:
                raise ValueError("模型字典不能为空")
            if preprocessor is None:
                raise ValueError("特征预处理器不能为None")
            if not training_features:
                raise ValueError("训练特征列表不能为空")
            if not target_features:
                raise ValueError("目标特征列表不能为空")
            if eval_df is None:
                eval_size = eval_size or self.eval_size
                if eval_size <= 0:
                    raise ValueError(f"评估样本数必须为正数，当前值：{eval_size}")
                db_conf = db_utils.db_config
                db_url = f"mysql+pymysql://{db_conf['user']}:{db_conf['password']}@" \
                         f"{db_conf['host']}:{db_conf['port']}/{db_conf['db']}?charset={db_conf['charset']}"
                engine = create_engine(db_url)
                query = text("""
                    SELECT * FROM t_prediction_parameters 
                    ORDER BY distance_from_entrance DESC 
                    LIMIT :limit
                """)
                with engine.connect() as conn:
                    eval_df = pd.read_sql(query, conn, params={"limit": eval_size})
                eval_df.columns = eval_df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')

            if eval_df.empty:
                msg = "未获取到评估样本，跳过评估"
                self._print_result(msg)
                return {"status": "warning", "message": msg}
            self._print_step(f"评估样本数：{len(eval_df)}")
            # 检查目标列
            missing_targets = [t for t in target_features if t not in eval_df.columns]
            if missing_targets:
                msg = f"评估数据缺少目标列：{missing_targets}，跳过评估"
                logger.warning(msg)
                self._print_result(msg)
                return {"status": "warning", "message": msg}
            # 特征对齐
            X_eval = eval_df[training_features]
            y_true = eval_df[target_features].values

            if list(X_eval.columns) != fitted_feature_order:
                logger.warning(f"评估特征顺序与训练不一致，重新排序")
                missing_cols = set(fitted_feature_order) - set(X_eval.columns)
                if missing_cols:
                    raise ValueError(f"评估数据缺少特征：{missing_cols}")
                X_eval = X_eval[fitted_feature_order]
            # 特征预处理+模型预测
            X_eval_proc = preprocessor.transform(X_eval)
            if not models:
                msg = "模型未训练，无法评估"
                logger.error(msg)
                return {"status": "error", "message": msg}
            # 检测算法类型
            algorithm_type = self._detect_algorithm_type(models)
            logger.debug(f"评估使用算法类型: {algorithm_type}")

            y_pred = []
            for target in target_features:
                if target not in models:
                    raise ValueError(f"未找到目标 {target} 的模型")
                model = models[target]

                # 根据算法类型进行预测
                if algorithm_type == "lightgbm":
                    pred = model.predict(X_eval_proc)
                elif algorithm_type == "xgboost":
                    # 将数据转换为XGBoost需要的格式
                    import xgboost as xgb
                    if hasattr(X_eval_proc, 'toarray'):
                        # 如果是稀疏矩阵，转换为稠密矩阵
                        X_eval_dense = X_eval_proc.toarray()
                    else:
                        X_eval_dense = X_eval_proc
                    dmatrix = xgb.DMatrix(X_eval_dense)
                    pred = model.predict(dmatrix)
                else:
                    raise ValueError(f"不支持的算法类型: {algorithm_type}")
                y_pred.append(pred.reshape(-1, 1))
            y_pred = np.hstack(y_pred)
            # 计算评估指标
            per_target_metrics = {}
            weighted_rmse_sum = 0.0
            for i, target in enumerate(target_features):
                weight = self.agg_weights[i] if i < len(self.agg_weights) else 1.0
                rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                per_target_metrics[target] = {
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "r2": round(r2, 4),
                    "weight": weight
                }
                weighted_rmse_sum += rmse * weight

            total_weight = sum(self.agg_weights[:len(target_features)]) or 1.0
            avg_rmse = round(weighted_rmse_sum / total_weight, 4)

            self._print_step("评估指标详情：")
            for target, metrics in per_target_metrics.items():
                self._print_step(f"  {target}：RMSE={metrics['rmse']}，MAE={metrics['mae']}，R²={metrics['r2']}")
            self._print_step(f"加权平均RMSE：{avg_rmse}")

            return {
                "status": "success",
                "avg_rmse": avg_rmse,
                "per_target": per_target_metrics,
                "sample_count": len(eval_df)
            }
        except Exception as e:
            logger.error(f"评估失败：{str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    def _performance_trigger_check(self, eval_history, baseline_rmse, current_sample_count=None):
        """
        增强的性能检查，考虑数据量因素
        """
        # 小数据模式：完全跳过性能检查
        if current_sample_count is not None and current_sample_count <= self.small_data_threshold:
            logger.info(
                f"小数据保护模式({current_sample_count}条数据)：跳过性能下降检查，"
                f"避免误报（阈值: {self.perf_drop_ratio:.0%}）"
            )
            return False
        # 数据量较少但超过10条：放宽检查条件
        if current_sample_count is not None and current_sample_count <= self.medium_data_threshold:
            adjusted_threshold = self.perf_drop_ratio * 2  # 双倍容忍度
            logger.info(
                f"中等数据保护模式({current_sample_count}条数据)：放宽性能检查，"
                f"调整阈值 {self.perf_drop_ratio:.0%} → {adjusted_threshold:.0%}"
            )
            # 临时调整阈值进行检查
            original_threshold = self.perf_drop_ratio
            self.perf_drop_ratio = adjusted_threshold
            try:
                result = self._original_performance_check(eval_history, baseline_rmse)
                return result
            finally:
                self.perf_drop_ratio = original_threshold
        # 正常数据量：使用原始检查逻辑
        return self._original_performance_check(eval_history, baseline_rmse)

    def _original_performance_check(self, eval_history, baseline_rmse):
        """
        原始性能检查逻辑（分离出来供小数据模式调用）
        """
        if not eval_history or len(eval_history) < self.perf_window + 1:
            return False
        recent_rmse = [h["avg_rmse"] for h in eval_history[-self.perf_window:]]
        recent_avg = sum(recent_rmse) / len(recent_rmse)
        baseline = baseline_rmse or recent_avg
        # 防止除以零
        if baseline == 0:
            return False
        drop_ratio = (recent_avg - baseline) / baseline
        logger.info(
            f"性能检测：基线RMSE={baseline:.4f}，最近{self.perf_window}次平均={recent_avg:.4f}，下降比例={drop_ratio:.2%}"
        )
        trigger = drop_ratio > self.perf_drop_ratio
        if trigger:
            logger.warning(f"性能下降触发：{drop_ratio:.2%} > 阈值{self.perf_drop_ratio:.0%}")
        return trigger

    def _rollback_if_worse(self, eval_result, baseline_rmse, rollback_callback):
        """私有方法：性能差于基线2倍阈值则回滚到上一版本"""
        if eval_result["status"] != "success":
            return
        current_rmse = eval_result["avg_rmse"]
        baseline = baseline_rmse
        if not current_rmse or not baseline:
            return
        drop_ratio = (current_rmse - baseline) / baseline
        if drop_ratio > self.perf_drop_ratio * 2:
            drop_ratio_str = f"{drop_ratio:.2%}" if drop_ratio is not None else "未知"
            threshold_str = f"{(self.perf_drop_ratio * 2):.2%}" if self.perf_drop_ratio is not None else "未知阈值"
            logger.warning(f"性能下降过多（{drop_ratio_str} > {threshold_str}），尝试回滚")

            if rollback_callback:
                rollback_res = rollback_callback(backup_index=-2)
                if rollback_res["success"]:
                    logger.info("回滚成功")
                else:
                    logger.error("回滚失败，需手动干预")

    def _detect_algorithm_type(self, models):
        """检测模型使用的算法类型"""
        if not models:
            return "unknown"
        # 检查第一个模型来判断算法类型
        first_model = next(iter(models.values()))
        # 通过类名判断
        class_name = type(first_model).__name__.lower()
        if 'booster' in class_name and hasattr(first_model, 'num_trees'):
            return "lightgbm"
        elif 'booster' in class_name and hasattr(first_model, 'get_dump'):
            return "xgboost"
        else:
            logger.warning(f"无法识别的模型类型: {class_name}")
            return "unknown"