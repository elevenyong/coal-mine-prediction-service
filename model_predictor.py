"""
煤矿瓦斯风险预测系统 - 模型预测模块
包含：模型预测、分源预测切换、结果格式化
"""
from loguru import logger
import xgboost as xgb
import numpy as np


from config_utils import timing_decorator


class ModelPredictor:
    """模型预测器"""

    def __init__(self):
        pass

    def _predict_model(self, model, X, algorithm_type="lightgbm"):
        """通用模型预测方法"""
        if algorithm_type == "lightgbm":
            return model.predict(X)
        elif algorithm_type == "xgboost":
            import xgboost as xgb
            # 将稀疏矩阵转换为XGBoost需要的格式
            if hasattr(X, 'toarray'):
                X_dense = X.toarray()
            else:
                X_dense = X
            return model.predict(xgb.DMatrix(X_dense))
        else:
            raise ValueError(f"不支持的算法类型: {algorithm_type}")

    def _detect_algorithm_type(self, models):
        """检测模型使用的算法类型（与ModelManager保持一致，避免误判）"""
        if not models:
            return "unknown"

        # 检查第一个模型来判断算法类型
        first_model = next(iter(models.values()))

        # 更精确的算法类型检测：优先判断模型“能力特征”，再用类名兜底
        if hasattr(first_model, 'save_model') and hasattr(first_model, 'num_trees'):
            # LightGBM Booster 通常具有 num_trees()
            return "lightgbm"
        elif hasattr(first_model, 'save_model') and hasattr(first_model, 'get_dump'):
            # XGBoost Booster 通常具有 get_dump()
            return "xgboost"
        else:
            # 通过类名判断（兜底）
            class_name = type(first_model).__name__.lower()
            if 'lgb' in class_name or 'lightgbm' in class_name:
                return "lightgbm"
            elif 'xgb' in class_name or 'xgboost' in class_name:
                return "xgboost"
            else:
                logger.warning(f"无法识别的模型类型: {class_name}")
                return "unknown"

    @timing_decorator
    def predict(self, data, models, preprocessor, training_features, target_features,
                fitted_feature_order, is_trained, file_lock, data_preprocessor, fault_calculator, db_utils):
        """
        公开方法：模型预测接口（支持批量，分源参数全则用分源结果，否则用模型预测）

        :param data: list[dict] / pandas.DataFrame，预测数据
        :param models: 模型字典
        :param preprocessor: 特征预处理器
        :param training_features: 训练特征列表
        :param target_features: 目标特征列表
        :param fitted_feature_order: 训练特征顺序
        :param is_trained: 模型是否已训练
        :param file_lock: 文件锁
        :param data_preprocessor: 数据预处理器
        :param fault_calculator: 断层计算器
        :param db_utils: 数据库工具
        :return: dict，预测结果
        """
        with file_lock:
            logger.info("模型预测开始（已移除瓦斯涌出量预测）")
            try:
                # Step 1: 检查模型状态
                if not is_trained or not models:
                    msg = "模型未训练或为空，无法预测"
                    logger.error(msg)
                    return {
                        "success": False,
                        "message": msg,
                        "predictions": None
                    }

                # Step 2: 数据预处理（预测模式）
                df = data_preprocessor.preprocess_data(
                    data, is_training=False,
                    fault_calculator=fault_calculator,
                    db_utils=db_utils
                )
                sample_count = len(df)
                logger.info(f"预测样本数：{sample_count}")

                # 预测阶段：必须以训练fit阶段的特征顺序为准；缺失特征列补0（尤其是历史/趋势特征）
                feature_cols = list(fitted_feature_order) if fitted_feature_order else list(training_features)

                # 防御：避免重复列名导致后续reindex异常
                if hasattr(df.columns, "duplicated") and df.columns.duplicated().any():
                    dup_cols = df.columns[df.columns.duplicated()].tolist()
                    logger.warning(f"预测数据存在重复列名（将保留首个并丢弃后续重复列）：{dup_cols}")
                    df = df.loc[:, ~df.columns.duplicated()]

                missing = [c for c in feature_cols if c not in df.columns]
                if missing:
                    logger.warning(f"预测输入缺失{len(missing)}个训练特征，将补0：{missing}")

                # 关键：reindex保证列齐全、顺序一致；缺失列统一补0
                X = df.reindex(columns=feature_cols, fill_value=0)
                # ---- 诊断：输出“最终进入模型”的特征行（21/24列）----
                try:
                    logger.info(f"【预测诊断】最终特征列数={len(feature_cols)}")
                    logger.info(f"【预测诊断】最终特征列名={feature_cols}")
                    # 单条预测时，直接打印第一行；多条则可打印前3条
                    logger.info(f"【预测诊断】最终特征首行={X.iloc[0].to_dict() if len(X) > 0 else {} }")
                except Exception as _e:
                    logger.warning(f"【预测诊断】特征行输出失败（已忽略）：{repr(_e)}")

                # Step 4: 特征预处理+模型预测
                X_proc = preprocessor.transform(X)
                predictions = {}

                # 关键修复：自动检测算法类型（避免硬编码导致XGBoost预测走错分支）
                algorithm_type = self._detect_algorithm_type(models)
                logger.debug(f"预测使用算法类型: {algorithm_type}")
                if algorithm_type == "unknown":
                    # 兜底：unknown时仍尝试按lightgbm执行（保持向后兼容），但给出明确警告
                    logger.warning("算法类型无法识别，预测将默认按lightgbm路径执行（请检查模型保存/加载逻辑）")
                    algorithm_type = "lightgbm"

                for target in target_features:
                    if target not in models:
                        raise ValueError(f"未找到目标 {target} 的预测模型")

                    # 使用统一预测方法
                    pred = self._predict_model(models[target], X_proc, algorithm_type=algorithm_type)
                    predictions[target] = [round(float(p), 4) for p in pred]

                # Step 5: 结果整理（移除了分源参数检查）
                result_list = []
                for i in range(sample_count):
                    pred_item = {tgt: predictions[tgt][i] for tgt in target_features}

                    # 补充标识信息（不再包含瓦斯涌出量）
                    for col in ["id", "working_face", "roadway", "distance_from_entrance"]:
                        if col in df.columns:
                            pred_item[col] = df.iloc[i][col]

                    result_list.append(pred_item)

                # Step 6: 返回标准化结果
                logger.info(f"预测完成，共 {sample_count} 条样本")
                return {
                    "success": True,
                    "message": "预测成功（已移除瓦斯涌出量预测）",
                    "predictions": result_list,
                    "target_features": target_features,
                    "sample_count": sample_count,
                    "note": "瓦斯涌出量请使用独立的/calculate_gas_emission_source接口计算"
                }
            except Exception as e:
                logger.error(f"预测失败：{str(e)}", exc_info=True)
                return {
                    "success": False,
                    "message": f"预测失败：{str(e)}",
                    "predictions": None,
                    "sample_count": len(df) if 'df' in locals() else 0
                }