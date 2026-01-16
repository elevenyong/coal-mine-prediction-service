"""
煤矿瓦斯风险预测系统 - 模型预测模块
包含：模型预测、分源预测切换、结果格式化
"""
import numpy as np
from loguru import logger
import pandas as pd
from sklearn.compose import ColumnTransformer


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
                # 根治修复：
                # 训练时 numeric_cols = [所有不在 base_categorical 的列]（见 ModelTrainer.create_preprocessor）
                # 因此预测时必须把“所有非 base_categorical 列”强制转数值，否则 SimpleImputer 内部会对 object 调 np.isnan 报错。
                try:
                    base_cat = []
                    if hasattr(data_preprocessor, "base_categorical") and isinstance(data_preprocessor.base_categorical,
                                                                                     list):
                        base_cat = data_preprocessor.base_categorical

                    # num_candidates：所有不在分类列表里的列，都视为数值列候选
                    num_candidates = [c for c in X.columns if c not in set(base_cat)]

                    # 诊断：先找出仍为 object 的“数值候选列”
                    obj_num_cols = [c for c in num_candidates if str(X[c].dtype) == "object"]
                    if obj_num_cols:
                        sample_preview = {}
                        for c in obj_num_cols[:10]:
                            try:
                                sample_preview[c] = X[c].dropna().astype(str).head(3).tolist()
                            except Exception:
                                sample_preview[c] = []
                        logger.warning(f"【预测诊断】发现数值候选列为object，将强制to_numeric：{obj_num_cols[:30]}")
                        logger.warning(f"【预测诊断】object列样例(前10列)：{sample_preview}")

                    # 强制数值化（即使原来是字符串数字，也会转成 float；无法转换的变 NaN）
                    for c in num_candidates:
                        try:
                            X[c] = pd.to_numeric(X[c], errors="coerce")
                        except Exception:
                            pass

                    # 统一清洗：inf/-inf -> NaN -> 0
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                except Exception as _e:
                    logger.warning(f"预测前数值强制转换失败（已忽略）：{repr(_e)}")

                # ===== 预测前：强制类型矫正（根治 isnan 报错）=====
                try:
                    base_cat = []
                    if hasattr(data_preprocessor, "base_categorical") and isinstance(data_preprocessor.base_categorical,
                                                                                     list):
                        base_cat = data_preprocessor.base_categorical

                    # 1) 分类列统一为字符串（避免被误当数值进入 isnan）
                    for c in base_cat:
                        if c in X.columns:
                            try:
                                X[c] = X[c].astype(str)
                            except Exception:
                                X[c] = X[c].apply(lambda v: "" if v is None else str(v))

                    # 2) 数值候选列：训练时一般是 “所有不在 base_categorical 的列”
                    num_candidates = [c for c in X.columns if c not in set(base_cat)]

                    # 2.1 datetime/timedelta -> float 时间戳（秒）
                    for c in num_candidates:
                        if c not in X.columns:
                            continue
                        try:
                            if pd.api.types.is_datetime64_any_dtype(X[c]) or pd.api.types.is_timedelta64_dtype(X[c]):
                                X[c] = (X[c].astype("int64") // 1_000_000_000).astype("float64")
                        except Exception:
                            # 兜底：变 NaN 后续补 0
                            X[c] = np.nan

                    # 2.2 其它全部强制 to_numeric（不可转 -> NaN）
                    for c in num_candidates:
                        if c not in X.columns:
                            continue
                        if pd.api.types.is_numeric_dtype(X[c]):
                            continue
                        try:
                            X[c] = pd.to_numeric(X[c], errors="coerce")
                        except Exception:
                            X[c] = np.nan

                    # 统一清洗
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                except Exception as e:
                    # 诊断必须可见：提升到 ERROR
                    logger.error(f"【预测诊断】类型矫正阶段失败：{repr(e)}")

                # ===== transform 失败时：打印“到底哪列不是数值” + “预处理器 numeric/cat 列定义”=====
                try:
                    X_proc = preprocessor.transform(X)
                except Exception as e:
                    try:
                        # 1) 打印所有列 dtype
                        dtypes_map = {c: str(X[c].dtype) for c in X.columns}
                        logger.error(f"【预测诊断】transform失败，X列dtype={dtypes_map}")
                        # 2) 打印首行（避免太大）
                        if len(X) > 0:
                            logger.error(f"【预测诊断】transform失败，X首行={X.iloc[0].to_dict()}")
                        # 3) 如果是 ColumnTransformer，打印 numeric/cat 实际列
                        if isinstance(preprocessor, ColumnTransformer) and hasattr(preprocessor, "transformers_"):
                            trans_cols = []
                            for name, trans, cols in preprocessor.transformers_:
                                # cols 可能是 list/array/slice
                                try:
                                    cols_list = list(cols) if not isinstance(cols, slice) else [
                                        f"slice({cols.start},{cols.stop},{cols.step})"]
                                except Exception:
                                    cols_list = [str(cols)]
                                trans_cols.append((name, cols_list[:50]))
                            logger.error(f"【预测诊断】ColumnTransformer列定义={trans_cols}")
                    except Exception as _e2:
                        logger.error(f"【预测诊断】transform失败后的诊断输出也失败：{repr(_e2)}")
                    raise  # 继续抛出，让上层返回原错误信息
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