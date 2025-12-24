"""
煤矿瓦斯风险预测系统 - 模型训练模块
包含：全量训练、增量训练、训练流程控制
依赖：lightgbm、scikit-learn
"""
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from loguru import logger
from sklearn.model_selection import train_test_split

from config_utils import error_handler_decorator


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config):
        self.config = config
        self.algorithm = config.get("Model", "algorithm", fallback="lightgbm")
        self.n_estimators = config.getint("Model", "n_estimators", fallback=100)
        self.increment_estimators = config.getint("Model", "increment_estimators", fallback=30)
        # 读取通用参数
        self.learning_rate = config.getfloat("Model", "learning_rate", fallback=0.05)  # 学习率（浮点数）
        # 读取LightGBM核心参数（从config.ini）
        self.num_leaves = config.getint("Model", "num_leaves", fallback=31)  # 叶子节点数（整数）
        self.reg_alpha = config.getfloat("Model", "reg_alpha", fallback=0.0)  # L1正则化（浮点数）
        self.reg_lambda = config.getfloat("Model", "reg_lambda", fallback=0.0)  # L2正则化（浮点数）

        # XGBoost特定参数
        self.max_depth = config.getint("Model", "max_depth", fallback=6)
        self.subsample = config.getfloat("Model", "subsample", fallback=0.8)
        self.colsample_bytree = config.getfloat("Model", "colsample_bytree", fallback=0.8)

        # 从配置读取过拟合判断阈值
        self.overfitting_threshold = config.getfloat("Model", "overfitting_threshold", fallback=1.5)
        self.underfitting_large_threshold = config.getfloat("Model", "underfitting_large_threshold", fallback=0.9)
        self.underfitting_small_threshold = config.getfloat("Model", "underfitting_small_threshold", fallback=1.2)
        # 存储训练诊断信息
        self.last_training_details = {}
        logger.info(f"模型训练器初始化完成，使用算法: {self.algorithm}")

    def _get_model_params(self):
        """获取当前算法的参数配置"""
        if self.algorithm == "lightgbm":
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': self.learning_rate,
                'num_leaves': self.num_leaves,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'verbosity': -1,
                'force_col_wise': True
            }
        elif self.algorithm == "xgboost":
            return {
                'objective': 'reg:squarederror',
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'verbosity': 0
            }
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    def _train_model(self, params, train_data, num_round, val_data=None, init_model=None):
        """通用模型训练方法"""
        if self.algorithm == "lightgbm":
            if val_data:
                return lgb.train(
                    params,
                    train_data,
                    num_boost_round=num_round,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'val'],
                    init_model=init_model,
                    callbacks=[lgb.log_evaluation(0)]
                )
            else:
                return lgb.train(
                    params,
                    train_data,
                    num_boost_round=num_round,
                    init_model=init_model,
                    callbacks=[lgb.log_evaluation(0)]
                )
        elif self.algorithm == "xgboost":
            eval_set = [(val_data, 'val')] if val_data else []
            return xgb.train(
                params,
                train_data,
                num_boost_round=num_round,
                evals=eval_set,
                xgb_model=init_model,
                verbose_eval=0
            )
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    def _create_dataset(self, X, y):
        """创建算法特定的数据集"""
        if self.algorithm == "lightgbm":
            return lgb.Dataset(X, label=y)
        elif self.algorithm == "xgboost":
            return xgb.DMatrix(X, label=y)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    def _predict_model(self, model, X):
        """模型预测方法"""
        if self.algorithm == "lightgbm":
            return model.predict(X)
        elif self.algorithm == "xgboost":
            if isinstance(X, xgb.DMatrix):
                return model.predict(X)
            else:
                return model.predict(xgb.DMatrix(X))
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    def _get_tree_count(self, model):
        """获取模型树数量"""
        if self.algorithm == "lightgbm":
            return model.num_trees()
        elif self.algorithm == "xgboost":
            return len(model.get_dump())
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    @error_handler_decorator
    def _full_train(self, X_proc, y, target_features):
        """
        私有方法：全量训练（清空现有模型，重新训练所有目标）

        :param X_proc: 预处理后的特征数据
        :param y: 目标值
        :param target_features: 目标特征列表
        :return: dict，训练好的模型字典
        """
        # 数据校验
        if X_proc is None or y is None:
            raise ValueError("训练数据X_proc或y不能为None")
        if len(X_proc) != len(y):
            raise ValueError(f"特征与目标数据长度不匹配：{len(X_proc)} vs {len(y)}")
        if not target_features:
            raise ValueError("目标特征列表不能为空")
        models = {}
        training_details = {
            'target_performance': {},
            'overfitting_diagnosis': {},
            'parameter_suggestions': [],
            'algorithm': self.algorithm
        }

        params = self._get_model_params()

        for i, target in enumerate(target_features):
            logger.info(f"全量训练开始 → 算法: {self.algorithm}, 目标: {target}, 预期树数量: {self.n_estimators}")

            # 使用交叉验证思路，但不减少训练数据
            if len(X_proc) > 20:  # 数据量足够时划分验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X_proc, y[:, i], test_size=0.2, random_state=42
                )
                use_validation = True
            else:
                # 数据量少时使用全部数据
                X_train, y_train = X_proc, y[:, i]
                X_val, y_val = X_proc, y[:, i]  # 用训练数据做"验证"
                use_validation = False

            train_data = self._create_dataset(X_train, y_train)
            val_data = self._create_dataset(X_val, y_val) if use_validation else None

            # 训练模型
            if use_validation:
                model = self._train_model(
                    params, train_data, self.n_estimators, val_data
                )
            else:
                model = self._train_model(
                    params, train_data, self.n_estimators
                )
            models[target] = model

            # 验证实际树数量是否与预期一致
            actual_trees = self._get_tree_count(model)
            if actual_trees != self.n_estimators:
                logger.warning(f"目标 {target} 实际树数量 ({actual_trees}) 与预期 ({self.n_estimators}) 不一致")

            # 计算性能指标（保守诊断）
            train_pred = self._predict_model(model, X_train)
            val_pred = self._predict_model(model, X_val) if use_validation else train_pred

            train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))

            # 过拟合诊断（保守判断）
            if use_validation:
                overfitting_ratio = val_rmse / train_rmse if train_rmse > 0 else 1.0
                is_overfitting = overfitting_ratio > self.overfitting_threshold
                is_underfitting = train_rmse > np.std(y_train) * self.underfitting_large_threshold
            else:
                # 数据量不足时，不进行过拟合诊断
                overfitting_ratio = 1.0
                is_overfitting = False
                is_underfitting = train_rmse > np.std(y_train) * self.underfitting_small_threshold

            training_details['target_performance'][target] = {
                'train_rmse': round(train_rmse, 4),
                'val_rmse': round(val_rmse, 4),
                'overfitting_ratio': round(overfitting_ratio, 2),
                'is_overfitting': is_overfitting,
                'is_underfitting': is_underfitting,
                'trees_count': actual_trees,
                'expected_trees': self.n_estimators,
                'use_validation': use_validation
            }

            logger.info(
                f"全量训练完成 → 算法: {self.algorithm}, 目标: {target}, "
                f"训练RMSE: {train_rmse:.4f}, 验证RMSE: {val_rmse:.4f}, "
                f"预期树数量: {self.n_estimators}, 实际树数量: {actual_trees}"
            )

        # 生成参数调整建议
        training_details['parameter_suggestions'] = self._generate_parameter_suggestions(
            training_details['target_performance'],
            getattr(self, 'config_filename', 'config.ini')
        )

        # 存储诊断信息供外部访问
        self.last_training_details = training_details

        return models

    @error_handler_decorator
    def _incremental_train(self, X_proc, y, target_features, existing_models):
        """
        私有方法：增量训练（基于现有模型追加训练）

        :param X_proc: 预处理后的特征数据
        :param y: 目标值
        :param target_features: 目标特征列表
        :param existing_models: 现有模型字典
        :return: dict，更新后的模型字典
        """
        if not existing_models:
            raise ValueError("增量训练需要非空的现有模型字典")
        if X_proc is None or y is None:
            raise ValueError("训练数据X_proc或y不能为None")
        if len(X_proc) == 0:
            raise ValueError("训练数据样本数量不能为0")

        params = self._get_model_params()
        training_details = {
            'target_performance': {},
            'overfitting_diagnosis': {},
            'parameter_suggestions': [],
            'algorithm': self.algorithm
        }

        for i, target in enumerate(target_features):
            init_model = existing_models.get(target)
            if init_model is None:
                raise ValueError(f"增量训练失败：目标 {target} 无现有模型")

            # 检查并打印模型状态
            initial_trees = self._get_tree_count(init_model) if init_model else 0
            logger.info(
                f"增量训练开始 → 算法: {self.algorithm}, 目标: {target}, 当前树数量: {initial_trees}, 计划追加: {self.increment_estimators}")

            # 如果现有模型树数量异常（≤1），使用 n_estimators 进行全量训练而不是 increment_estimators
            if initial_trees <= 1:
                logger.warning(f"目标 {target} 的现有模型树数量异常（{initial_trees}），使用全量训练代替增量训练（n_estimators={self.n_estimators}）")
                # 使用全量训练逻辑，但使用当前数据
                if len(X_proc) > 20:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_proc, y[:, i], test_size=0.2, random_state=42
                    )
                    use_validation = True
                else:
                    X_train, y_train = X_proc, y[:, i]
                    X_val, y_val = X_proc, y[:, i]
                    use_validation = False

                train_data = self._create_dataset(X_train, y_train)
                val_data = self._create_dataset(X_val, y_val) if use_validation else None

                # 使用 n_estimators 而不是 increment_estimators 进行全量训练
                if use_validation:
                    model = self._train_model(
                        params, train_data, self.n_estimators, val_data
                    )
                else:
                    model = self._train_model(
                        params, train_data, self.n_estimators
                    )
                existing_models[target] = model
                final_trees = self._get_tree_count(model)
                actual_increment = final_trees  # 因为是全量训练，所以增量等于最终树数量
            else:
                # 正常的增量训练逻辑
                train_data = self._create_dataset(X_proc, y[:, i])
                model = self._train_model(
                    params, train_data, self.increment_estimators, init_model=init_model
                )
                existing_models[target] = model

                # 验证树数量是否正确累加
                final_trees = self._get_tree_count(model)
                actual_increment = final_trees - initial_trees

            logger.info(
                f"增量训练完成 → 算法: {self.algorithm}, 目标: {target}, 追加树数量: {actual_increment}, 累计树数量: {final_trees}")

            # 计算性能指标（使用训练数据）
            train_pred = self._predict_model(model, X_proc)
            train_rmse = np.sqrt(np.mean((y[:, i] - train_pred) ** 2))

            # 增量训练不进行过拟合诊断（数据不足）
            training_details['target_performance'][target] = {
                'train_rmse': round(train_rmse, 4),
                'val_rmse': round(train_rmse, 4),  # 与训练集相同
                'overfitting_ratio': 1.0,
                'is_overfitting': False,
                'is_underfitting': train_rmse > np.std(y[:, i]) * 1.2,
                'trees_count': final_trees,
                'incremental_trees': actual_increment,
                'use_validation': False,
                'initial_trees': initial_trees,
                'train_mode': 'full' if initial_trees <= 1 else 'incremental'  # 记录训练模式
            }

            logger.info(
                f"目标 {target} → 训练模式: {'全量训练' if initial_trees <= 1 else '增量训练'}, "
                f"追加树数量: {actual_increment}, "
                f"累计树数量: {final_trees}, "
                f"训练RMSE: {train_rmse:.4f}"
            )

        # 生成参数调整建议
        training_details['parameter_suggestions'] = self._generate_parameter_suggestions(
            training_details['target_performance'],
            getattr(self, 'config_filename', 'config.ini')
        )

        # 存储诊断信息
        self.last_training_details = training_details

        return existing_models

    def _generate_parameter_suggestions(self, target_performance, config_filename="config.ini"):
        """
        生成参数调整建议基于模型性能诊断

        :param target_performance: 各目标性能指标
        :return: list, 参数调整建议
        """
        suggestions = []

        algorithm_specific_advice = {
            'lightgbm': [
                "  • 调整 num_leaves（当前值：{}）控制模型复杂度".format(self.num_leaves),
                "  • 调整 reg_alpha（当前值：{}）控制L1正则化".format(self.reg_alpha),
                "  • 调整 reg_lambda（当前值：{}）控制L2正则化".format(self.reg_lambda)
            ],
            'xgboost': [
                "  • 调整 max_depth（当前值：{}）控制树深度".format(self.max_depth),
                "  • 调整 subsample（当前值：{}）控制样本采样".format(self.subsample),
                "  • 调整 colsample_bytree（当前值：{}）控制特征采样".format(self.colsample_bytree)
            ]
        }

        # 统计过拟合和欠拟合情况
        overfitting_targets = [t for t, perf in target_performance.items()
                               if perf.get('is_overfitting', False)]
        underfitting_targets = [t for t, perf in target_performance.items()
                                if perf.get('is_underfitting', False)]

        if overfitting_targets:
            suggestions.extend([
                f"📈 检测到过拟合现象，建议调整{algorithm_specific_advice.get(self.algorithm, [])}:"
            ])
            suggestions.extend(algorithm_specific_advice.get(self.algorithm, []))
            suggestions.append("  • 减小 learning_rate（当前值：{}）并增加 n_estimators".format(self.learning_rate))

        if underfitting_targets:
            suggestions.extend([
                f"📉 检测到欠拟合现象，建议调整{algorithm_specific_advice.get(self.algorithm, [])}:"
            ])
            suggestions.extend(algorithm_specific_advice.get(self.algorithm, []))
            suggestions.extend([
                "  • 增加 learning_rate（当前值：{}）加速学习".format(self.learning_rate),
                "  • 增加 n_estimators（当前值：{}）延长训练".format(self.n_estimators)
            ])

        if not overfitting_targets and not underfitting_targets:
            suggestions.extend([
                "✅ 模型拟合状态良好，当前参数配置合理",
                f"💡 可尝试微调 learning_rate 或 {self.algorithm} 特定参数进一步优化性能"
            ])

        # 添加通用建议
        suggestions.extend([
            f"🔧 参数调整位置：{config_filename} -> [Model] section",
            f"🎯 当前使用算法：{self.algorithm}",
            "💾 修改配置后重启服务生效"
        ])

        return suggestions

    def get_last_training_diagnostics(self):
        """
        新增方法：获取最后一次训练的诊断信息
        保持向后兼容
        """
        return self.last_training_details

    def create_preprocessor(self, X, base_categorical):
        """
        创建特征预处理器

        :param X: 特征数据
        :param base_categorical: 分类特征列表
        :return: 预处理器和特征顺序
        """
        numeric_cols = [col for col in X.columns if col not in base_categorical]
        categorical_cols = [col for col in X.columns if col in base_categorical]

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        X_proc = preprocessor.fit_transform(X)
        logger.debug(f"特征预处理完成：数值特征 {len(numeric_cols)} 个，分类特征 {len(categorical_cols)} 个")

        return preprocessor, X_proc, X.columns.tolist()

    def transform_features(self, preprocessor, X, fitted_feature_order):
        """
        转换特征数据

        :param preprocessor: 预处理器
        :param X: 特征数据
        :param fitted_feature_order: 训练时的特征顺序
        :return: 转换后的特征数据
        """
        if list(X.columns) != fitted_feature_order:
            logger.warning(f"特征顺序与训练不一致，重新排序")
            missing_cols = set(fitted_feature_order) - set(X.columns)
            if missing_cols:
                raise ValueError(f"数据缺少特征：{missing_cols}")
            X = X[fitted_feature_order]

        return preprocessor.transform(X)