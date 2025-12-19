"""
煤矿瓦斯风险预测系统 - 主模型类
整合所有模块，提供统一的模型接口
"""
import os
from datetime import datetime

import pandas as pd
from filelock import FileLock
from loguru import logger
import configparser
# 导入各功能模块
from config_utils import ConfigUtils
from fault_calculator import FaultCalculator
from regional_measure_calculator import RegionalMeasureCalculator
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from model_predictor import ModelPredictor
from model_manager import ModelManager
from db_utils import DBUtils
class CoalMineRiskModel:
    """
    煤矿瓦斯风险预测模型（LightGBM多目标回归）
    使用组合模式而非继承来整合各功能模块
    """

    def __init__(self, config_path="config.ini",
                 fault_calculator=None,
                 regional_calculator=None,
                 data_preprocessor=None,
                 model_trainer=None,
                 model_evaluator=None,
                 model_predictor=None,
                 model_manager=None,
                 db_utils=None):
        """
        模型初始化入口：读取配置→初始化目录→加载参数→初始化依赖→加载已有模型
        """
        # Step 1: 初始化配置
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding="utf-8")
        logger.info(f"成功加载配置文件：{config_path}")
        # Step 2: 初始化各功能模块
        self.config_utils = ConfigUtils(config_path)
        self.fault_calculator = fault_calculator or FaultCalculator(config_path)
        self.regional_calculator = regional_calculator or RegionalMeasureCalculator(config_path)
        self.data_preprocessor = data_preprocessor or DataPreprocessor(config_path)
        self.model_trainer = model_trainer or ModelTrainer(self.config)
        self.model_trainer.config_filename = os.path.basename(config_path)
        self.model_evaluator = model_evaluator or ModelEvaluator(config_path)
        self.model_predictor = model_predictor or ModelPredictor()
        # Step 3: 初始化模型管理器
        self.model_dir = self.config_utils._get_config_value("Model", "model_dir", "models")
        self.model_manager = model_manager or ModelManager(self.model_dir)
        # Step 4: 加载训练核心参数
        self.full_train_threshold = self.config_utils._get_config_value("Model", "full_train_threshold", 6, is_int=True)
        self.min_train_samples = self.config_utils._get_config_value("Model", "min_train_samples", 6, is_int=True)
        # Step 5: 模型核心状态初始化
        self.models = {}
        self.preprocessor = None
        self.training_features = None
        self.is_trained = False
        self.total_samples = 0
        self.eval_history = []
        self.training_stats = []
        self.baseline_rmse = None
        self._fitted_feature_order = None
        # Step 6: 初始化数据库工具与跨进程锁
        self.db = DBUtils(config_path=config_path)
        self.file_lock = FileLock(self.model_manager.lock_file_path)
        logger.info(f"跨进程锁初始化完成，锁文件路径：{self.model_manager.lock_file_path}")
        # Step 7: 加载已有模型与同步数据库样本数
        self._load_model()
        try:
            self.total_samples = self.model_manager.get_total_samples_from_db(self.db)
        except Exception as e:
            self.total_samples = 0
            logger.warning(f"同步数据库样本数失败：{str(e)}，初始化为0")
        # Step 8: 控制台输出初始化结果
        self._print_header("模型初始化完成")
        self.current_config = config_path  # 记录当前使用的配置文件
        # ============ 20251218 新增：进尺特征状态初始化 ============
        self.mining_advance_enabled = getattr(self.data_preprocessor, 'enable_cumulative_advance', False)
        logger.info(f"进尺特征状态：{'已启用' if self.mining_advance_enabled else '已禁用'}")
        # ============ 新增结束 ============
        logger.info(f"当前配置文件：{self.current_config}")
        # 添加缺失的属性初始化
        self.fixed_evaluation_set = None  # 固定评估数据集
        if self.config.getboolean("Logging", "verbose_console", fallback=True):
            # 确保这些属性存在
            base_categorical_count = len(getattr(self.data_preprocessor, 'base_categorical', []))
            base_numeric_count = len(getattr(self.data_preprocessor, 'base_numeric', []))
            target_features_count = len(getattr(self.data_preprocessor, 'target_features', []))
            algorithm = getattr(self.model_trainer, 'algorithm', 'lightgbm')
            print(f"├─ 分类特征数量：{base_categorical_count}")
            print(f"├─ 数值特征数量：{base_numeric_count}")
            print(f"├─ 预测目标数量：{target_features_count}")
            print(f"├─ 累计样本数：{self.total_samples}")
            print(f"├─ 使用算法：{algorithm}")
            print(f"└─ 模型状态：{'已训练' if self.is_trained else '未训练'}")
            print("=" * 60)
            # ============ 20251218新增：输出进尺特征状态 ============
            if hasattr(self.data_preprocessor, 'enable_cumulative_advance'):
                print(f"├─ 进尺特征：{'已启用' if self.data_preprocessor.enable_cumulative_advance else '已禁用'}")
            # ============ 新增结束 ============

    def reload_config(self, new_config_path=None, reload_database=False):
        """
        动态重载配置（不重启服务）
        保持原有模型状态，只更新配置参数

        :param new_config_path: str，新配置文件路径，默认None（使用当前路径）
        :param reload_database: bool，是否重载数据库配置，默认False（避免不必要的连接重建）
        :return: bool，重载是否成功
        """
        try:
            logger.info("开始动态重载系统配置")
            # Step 1: 备份关键状态
            current_models = self.models.copy() if self.models else {}
            current_total_samples = self.total_samples
            current_training_stats = self.training_stats.copy() if hasattr(self, 'training_stats') else []
            current_eval_history = self.eval_history.copy() if hasattr(self, 'eval_history') else []
            current_baseline_rmse = self.baseline_rmse if hasattr(self, 'baseline_rmse') else None
            current_is_trained = self.is_trained
            current_preprocessor = self.preprocessor
            current_training_features = self.training_features
            current_fitted_feature_order = self._fitted_feature_order
            current_modules = {
                'fault_calculator': self.fault_calculator,
                'regional_calculator': self.regional_calculator,
                'data_preprocessor': self.data_preprocessor,
                'model_trainer': self.model_trainer,
                'model_evaluator': self.model_evaluator,
                'model_manager': self.model_manager,
                'db': self.db
            }
            # ============ 20251218新增：重载进尺配置 ============
            # 重新初始化数据预处理器（会重新加载进尺配置）
            self.data_preprocessor = DataPreprocessor(self.config_path)

            # 更新进尺特征状态
            self.mining_advance_enabled = getattr(self.data_preprocessor, 'enable_cumulative_advance', False)
            logger.info(f"进尺特征配置重载：{'已启用' if self.mining_advance_enabled else '已禁用'}")
            # ============ 20251218新增结束 ============
            # Step 2: 更新配置文件路径（如果提供了新路径）
            if new_config_path:
                self.config_path = new_config_path
                logger.info(f"切换到新配置文件: {new_config_path}")
            # Step 3: 重新读取配置
            merged_config = configparser.ConfigParser()
            # 首先读取基础配置 config.ini
            merged_config.read("config.ini", encoding="utf-8")
            logger.debug("基础配置文件 config.ini 已加载")

            # 然后读取阶段配置（如果有），它会覆盖基础配置中的相同项
            if self.config_path and self.config_path != "config.ini":
                stage_config_read = merged_config.read(self.config_path, encoding="utf-8")
                if stage_config_read:
                    logger.debug(f"阶段配置文件 {self.config_path} 已加载并合并")
                else:
                    logger.warning(f"阶段配置文件 {self.config_path} 读取失败，仅使用基础配置")
            # 更新当前配置对象
            self.config = merged_config
            logger.debug("配置文件合并完成")
            # Step 4: 重新初始化配置工具和各模块,使用合并后的配置对象来初始化各模块
            self.config_utils = ConfigUtils(self.config_path)
            # 重新初始化各功能模块
            self.fault_calculator = FaultCalculator(self.config_path)
            self.regional_calculator = RegionalMeasureCalculator(self.config_path)
            self.data_preprocessor = DataPreprocessor(self.config_path)
            self.model_trainer = ModelTrainer(self.config)
            self.model_evaluator = ModelEvaluator(self.config_path)
            self.model_manager = ModelManager(self.model_dir)
            # Step 5: 条件性重载数据库配置
            if reload_database:
                logger.info("重载数据库配置（将重建数据库连接）")
                db_reload_success = self.db.reload_config(self.config_path)
                if not db_reload_success:
                    logger.error("数据库配置重载失败，但继续其他配置重载")
            else:
                logger.info("跳过数据库配置重载（使用现有连接）")
            # Step 6: 重新加载模型核心参数
            self.full_train_threshold = self.config_utils._get_config_value("Model", "full_train_threshold", 6,is_int=True)
            self.min_train_samples = self.config_utils._get_config_value("Model", "min_train_samples", 6,is_int=True)
            # Step 7: 恢复关键状态
            self.models = current_models
            self.total_samples = current_total_samples
            self.training_stats = current_training_stats
            self.eval_history = current_eval_history
            self.baseline_rmse = current_baseline_rmse
            self.is_trained = current_is_trained
            self.preprocessor = current_preprocessor
            self.training_features = current_training_features
            self._fitted_feature_order = current_fitted_feature_order
            # Step 8: 同步预处理器状态
            if hasattr(self.data_preprocessor, 'is_trained'):
                self.data_preprocessor.is_trained = current_is_trained
            if hasattr(self.data_preprocessor, 'training_features') and self.training_features:
                self.data_preprocessor.training_features = self.training_features
            if hasattr(self, 'model_trainer'):
                self.model_trainer.config_filename = os.path.basename(self.config_path)
            logger.info("配置动态重载完成，所有模块已更新")
            self._print_result("配置动态重载成功")
            # 输出新的关键参数值
            if self.config.getboolean("Logging", "verbose_console", fallback=True):
                print(f"├─ 新的全量训练阈值: {self.full_train_threshold}")
                print(f"├─ 新的最小样本数: {self.min_train_samples}")
                print(f"├─ 新的树数量: {self.model_trainer.n_estimators}")
                print(f"├─ 新的学习率: {self.model_trainer.learning_rate}")
                if reload_database:
                    print(f"└─ 数据库配置已重载")
                else:
                    print(f"└─ 数据库配置未重载（使用现有连接）")
            return True
        except Exception as e:
            logger.error(f"配置动态重载失败，正在恢复状态：{str(e)}", exc_info=True)
            # 恢复模块状态
            self.fault_calculator = current_modules['fault_calculator']
            self.regional_calculator = current_modules['regional_calculator']
            self.data_preprocessor = current_modules['data_preprocessor']
            self.model_trainer = current_modules['model_trainer']
            self.model_evaluator = current_modules['model_evaluator']
            self.model_manager = current_modules['model_manager']
            self.db = current_modules['db']
            self._print_result(f"配置重载失败：{str(e)}")
            return False
    def _print_header(self, title):
        """委托给config_utils"""
        self.config_utils._print_header(title)

    def _print_step(self, msg):
        """委托给config_utils"""
        self.config_utils._print_step(msg)

    def _print_result(self, msg):
        """委托给config_utils"""
        self.config_utils._print_result(msg)

    def _load_model(self):
        """加载已有模型组件"""
        self._print_header("加载已有模型")
        try:
            # 确保target_features属性存在
            target_features = getattr(self.data_preprocessor, 'target_features', [])
            if not target_features:
                logger.warning("目标特征列表为空，无法加载模型")
                self.is_trained = False
                return
            (self.preprocessor, self.training_features,
             self.models, self.is_trained) = self.model_manager.load_model(target_features)
            if self.is_trained:
                self.data_preprocessor.is_trained = True
                self.data_preprocessor.training_features = self.training_features
                self._print_step("模型加载成功")
            else:
                self._print_step("模型加载失败或未训练，需从头训练")
        except Exception as e:
            self.is_trained = False
            logger.error(f"加载模型失败：{str(e)}", exc_info=True)
            self._print_step(f"模型加载失败：{str(e)}，需从头训练")
        finally:
            if self.config.getboolean("Logging", "verbose_console", fallback=True):
                print("-" * 60)

    def calculate_fault_influence_strength(self, data):
        """计算断层影响系数（委托给FaultCalculator）"""
        return self.fault_calculator.calculate_fault_influence_strength(data, self.db)

    def calculate_regional_measure_strength(self, measures):
        """计算区域措施强度（委托给RegionalMeasureCalculator）"""
        return self.regional_calculator.calculate_regional_measure_strength(measures)

    def _preprocess_data(self, data, is_training=True):
        """数据预处理（委托给DataPreprocessor）"""
        if is_training:
            df, training_features = self.data_preprocessor.preprocess_data(
                data, is_training, self.fault_calculator, self.db
            )
            self.training_features = training_features
            return df
        else:
            return self.data_preprocessor.preprocess_data(
                data, is_training, self.fault_calculator, self.db
            )

    def _print_training_diagnosis(self):
        """
        新增：打印训练诊断结果和参数建议
        不修改原有训练流程
        """
        training_details = self.model_trainer.get_last_training_diagnostics()
        if not training_details:
            return
        target_performance = training_details.get('target_performance', {})
        suggestions = training_details.get('parameter_suggestions', [])
        self._print_header("模型训练诊断结果")
        # 打印各目标性能
        for target, perf in target_performance.items():
            status = "✅ 良好"
            if perf.get('is_overfitting', False):
                status = "⚠️ 过拟合"
            elif perf.get('is_underfitting', False):
                status = "📉 欠拟合"
            validation_note = "(验证集)" if perf.get('use_validation', False) else "(训练集)"
            self._print_step(
                f"{target}: 训练RMSE={perf['train_rmse']}, "
                f"{validation_note}RMSE={perf['val_rmse']}, "
                f"过拟合比率={perf['overfitting_ratio']} {status}"
            )
        # 打印参数建议
        if suggestions:
            self._print_header("参数调整建议")
            for suggestion in suggestions:
                self._print_step(suggestion)

    # ============ 20251218新增：进尺特征辅助方法 ============
    def _validate_and_log_mining_features(self, df):
        """
        验证和记录进尺特征信息
        :param df: 预处理后的DataFrame
        """
        try:
            # 检查进尺特征是否存在
            required_features = ['cumulative_advance', 'effective_exposure_distance']
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                logger.warning(f"进尺特征缺失：{missing_features}")
            else:
                # 记录进尺特征统计
                cum_stats = {
                    'min': df['cumulative_advance'].min(),
                    'max': df['cumulative_advance'].max(),
                    'mean': df['cumulative_advance'].mean()
                }
                exp_stats = {
                    'min': df['effective_exposure_distance'].min(),
                    'max': df['effective_exposure_distance'].max(),
                    'mean': df['effective_exposure_distance'].mean()
                }
                logger.info(
                    f"进尺特征统计：累计进尺[{cum_stats['min']:.1f}~{cum_stats['max']:.1f}], "
                    f"均值={cum_stats['mean']:.1f}; "
                    f"有效暴露距离[{exp_stats['min']:.1f}~{exp_stats['max']:.1f}], "
                    f"均值={exp_stats['mean']:.1f}"
                )
                # 控制台输出
                if self.config.getboolean("Logging", "verbose_console", fallback=True):
                    print("├─ 进尺特征统计：")
                    print(
                        f"│  ├─ 累计进尺：{cum_stats['min']:.1f}~{cum_stats['max']:.1f}米，均值={cum_stats['mean']:.1f}米")
                    print(
                        f"│  ├─ 有效暴露距离：{exp_stats['min']:.1f}~{exp_stats['max']:.1f}米，均值={exp_stats['mean']:.1f}米")
        except Exception as e:
            logger.warning(f"进尺特征验证失败：{str(e)}")
    # ============ 20251218新增结束 ============
    def train(self, data, epochs=1):
        """
        公开方法：模型训练接口
        """
        with self.file_lock:
            self._print_header("模型训练开始")
            train_start = datetime.now()
            initial_samples = self.total_samples
            db_conn = None
            db_trans = None
            custom_create_time = None
            saved_count = 0
            try:
                # Step 1: 数据预处理
                df = self._preprocess_data(data, is_training=True)
                # ============ 20251218 新增：时空特征验证与记录 ============
                if hasattr(self.data_preprocessor, 'spatiotemporal_extractor') and \
                        self.data_preprocessor.spatiotemporal_extractor:
                    # 获取新增的时空特征信息
                    new_features = self.data_preprocessor.spatiotemporal_extractor.get_new_feature_names()
                    if new_features:
                        logger.info(f"本次训练使用了 {len(new_features)} 个时空特征")

                        # 按类别统计
                        categories = self.data_preprocessor.spatiotemporal_extractor.get_all_new_feature_categories()
                        for category, features in categories.items():
                            if features:
                                logger.debug(f"  {category}: {len(features)}个特征")

                        # 控制台输出
                        if self.config.getboolean("Logging", "verbose_console", fallback=True):
                            print(f"├─ 时空特征使用情况：")
                            for category, features in categories.items():
                                if features:
                                    print(f"│  ├─ {category}: {len(features)}个")
                # ============ 20251218 新增结束：时空特征验证与记录 ============
                if self.mining_advance_enabled:
                    self._validate_and_log_mining_features(df)
                # ============ 20251218新增结束 ============
                if len(df) < self.min_train_samples:
                    msg = f"样本数 {len(df)} < 最小训练样本数 {self.min_train_samples}，跳过训练"
                    logger.warning(msg)
                    self._print_result(msg)
                    return {
                        "status": "warning",
                        "message": msg,
                        "training_stats": {"processed_samples": len(df), "training_performed": False}
                    }
                # ============ 新增：自动配置切换逻辑 ============
                # 判断是否初次训练（模型未训练且数据库样本数为0）
                is_initial_training = not self.is_trained and initial_samples == 0
                # 判断数据量大小（使用全量训练阈值作为判断标准）
                is_large_data = len(df) >= self.full_train_threshold
                # 自动配置切换决策
                if is_initial_training and is_large_data:
                    # 初次大量数据训练 → 使用 phase1 配置
                    target_config = "config_phase1.ini"
                    reason = "初次大量数据训练"
                elif self.is_trained and not is_large_data:
                    # 后续少量数据增量训练 → 使用 phase2 配置
                    target_config = "config_phase2.ini"
                    reason = "少量数据增量训练"
                else:
                    # 其他情况保持当前配置
                    target_config = None
                    reason = "保持当前配置"
                # 执行配置切换（如果需要）
                if target_config and getattr(self, 'current_config', None) != target_config:
                    logger.info(f"自动配置切换：{reason} → {target_config}")
                    success = self.reload_config(target_config)
                    if success:
                        self.current_config = target_config
                        self.model_trainer.config_filename = os.path.basename(target_config)
                        self._print_step(f"✅ 配置已切换：{target_config}（{reason}）")
                    else:
                        logger.error(f"配置切换失败：{target_config}")
                        self._print_step(f"❌ 配置切换失败：{target_config}")
                elif target_config:
                    logger.debug(f"配置已是最新：{target_config}")
                else:
                    logger.debug(f"无需切换配置：{reason}")
                # ============ 自动配置切换逻辑结束 ============
                # Step 2: 开启数据库事务
                db_conn = self.db._get_connection()
                db_trans = db_conn.begin()
                logger.info("数据库事务已开启，准备保存训练数据")
                # Step 3: 保存数据到数据库
                custom_create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                saved_count = self.db.save_training_data(
                    df=df,
                    conn=db_conn,
                    trans=db_trans,
                    custom_create_time=custom_create_time
                )
                if saved_count == 0:
                    raise ValueError("数据保存到数据库失败（保存条数为0）")
                logger.info(f"事务中插入 {saved_count} 条训练数据（未提交）")

                # Step 4: 构建训练集
                if not self.training_features:
                    raise ValueError("训练特征列表为空，请检查数据预处理")
                X = df[self.training_features]
                y = df[self.data_preprocessor.target_features].values
                self._fitted_feature_order = X.columns.tolist()
                # 记录特征信息
                logger.info(
                    f"训练特征数量: {len(self.training_features)}，包含时空特征: {any('neighbor_' in f or 'decay' in f for f in self.training_features)}")
                # Step 5: 特征预处理
                self.preprocessor, X_proc, _ = self.model_trainer.create_preprocessor(X, self.data_preprocessor.base_categorical)

                # Step 6: 判断训练模式
                current_total = initial_samples + len(df)
                threshold_hit = (current_total % self.full_train_threshold == 0)
                # 传递数据量信息的性能检查
                perf_trigger = self.model_evaluator._performance_trigger_check(
                    self.eval_history,
                    self.baseline_rmse,
                    len(df)  # 传递当前训练数据量
                )
                if perf_trigger:
                    do_full_train = True
                    reason = "性能下降触发"
                elif threshold_hit:
                    do_full_train = True
                    reason = "样本数达阈值触发"
                elif not self.is_trained:
                    do_full_train = True
                    reason = "首次训练"
                else:
                    do_full_train = False
                    reason = "增量训练"
                self._print_step(f"训练模式判断：{reason}")
                # Step 7: 执行训练
                if do_full_train:
                    self.models = self.model_trainer._full_train(X_proc, y, self.data_preprocessor.target_features)
                else:
                    self.models = self.model_trainer._incremental_train(X_proc, y, self.data_preprocessor.target_features, self.models)
                # Step 8: 提交事务+更新状态
                db_trans.commit()
                logger.info(f"事务提交成功，{saved_count} 条数据已持久化")
                self.total_samples = self.model_manager.get_total_samples_from_db(self.db)
                new_samples = self.total_samples - initial_samples
                self._print_step(f"样本数更新：新增 {new_samples} 条，累计 {self.total_samples} 条")
                # Step 9: 保存模型+标记训练状态
                self.model_manager.save_model(self.models, self.preprocessor, self.training_features, self.data_preprocessor.target_features)
                self.is_trained = True
                # 同步状态到预处理器
                self.data_preprocessor.is_trained = True
                self.data_preprocessor.training_features = self.training_features
                # Step 10: 训练后评估（小数据特殊处理）
                if len(df) <= 5:
                    # 极少量数据：跳过评估，避免不准确的RMSE影响基线
                    eval_result = {
                        "status": "skipped",
                        "message": f"数据量过少({len(df)}条)，跳过评估避免误判",
                        "avg_rmse": None
                    }
                    agg_rmse = None
                    logger.info(f"小数据训练({len(df)}条)：跳过性能评估")
                else:
                    # 正常数据量：执行评估
                    eval_result = self.evaluate_model()
                    agg_rmse = eval_result.get("avg_rmse")

                # Step 11: 智能设置性能基线
                if agg_rmse and self.baseline_rmse is None:
                    # 首次设置基线
                    self.baseline_rmse = agg_rmse
                    logger.info(f"设置初始性能基线: {agg_rmse:.4f}")
                elif agg_rmse and len(df) >= 10:
                    # 只有数据量足够时才更新基线，避免小数据干扰
                    self.baseline_rmse = agg_rmse
                    logger.info(f"更新性能基线: {agg_rmse:.4f}")

                # Step 12: 性能回滚检查（添加小数据保护）
                # 修正：使用正确的变量名 agg_rmse 而不是 current_rmse
                if (eval_result["status"] == "success" and
                        agg_rmse is not None and  # 修正：使用 agg_rmse
                        self.baseline_rmse is not None and
                        len(df) > 10):  # 只有数据量>10时才检查性能下降
                    drop_ratio = (agg_rmse - self.baseline_rmse) / self.baseline_rmse  # 修正：使用正确的变量
                    # 添加详细的性能诊断日志
                    logger.info(f"=== 性能检查诊断 ===")
                    logger.info(f"训练数据: {len(df)}条, 当前RMSE: {agg_rmse:.4f}")  # 修正：使用 agg_rmse
                    logger.info(f"基线RMSE: {self.baseline_rmse:.4f}, 下降比例: {drop_ratio:.2%}")
                    logger.info(f"阈值: {self.model_evaluator.perf_drop_ratio * 2:.2%}")
                    if drop_ratio > self.model_evaluator.perf_drop_ratio * 2:
                        logger.warning(f"性能下降过多（{drop_ratio:.2%}），尝试回滚")
                        rollback_res = self.rollback_model(backup_index=-2)
                        if rollback_res["success"]:
                            # 重新加载模型
                            self._load_model()
                            # 更新基线
                            if self.eval_history and len(self.eval_history) > 1:
                                self.baseline_rmse = self.eval_history[-2]["avg_rmse"]
                            logger.info(f"回滚成功，基线RMSE恢复为：{self.baseline_rmse}")
                else:
                    skip_reason = []
                    if eval_result["status"] != "success":
                        skip_reason.append("评估失败")
                    if agg_rmse is None:
                        skip_reason.append("无RMSE数据")
                    if self.baseline_rmse is None:
                        skip_reason.append("无基线数据")
                    if len(df) <= 10:
                        skip_reason.append("小数据训练")
                    logger.info(f"跳过性能回滚检查: {', '.join(skip_reason)}")
                # Step 13: 如果是大数据量初始训练，设置固定评估集
                if len(df) >= 100 and not hasattr(self, 'fixed_evaluation_set'):
                    logger.info("大数据量训练，设置固定评估集")
                    self.set_fixed_evaluation_set(df, size=50)
                # Step 12: 记录训练历史到数据库
                train_duration = (datetime.now() - train_start).total_seconds()
                # 监控调用
                from performance_monitor import global_monitor
                global_monitor.record_training_session(
                    train_mode=reason,  # 从原有变量获取
                    sample_count=len(df),  # 从原有变量获取
                    duration=train_duration,
                    rmse=agg_rmse  # 从原有变量获取
                )
                training_record = {
                    "sample_count": len(df),
                    "total_samples": self.total_samples,
                    "train_mode": reason,
                    "status": "success",
                    "message": f"训练完成（{reason}），RMSE：{agg_rmse:.4f}" if agg_rmse else f"训练完成（{reason}），RMSE：未计算",
                    "duration": train_duration,
                    "train_time": datetime.now()
                }
                record_id = self.db.insert_training_record(training_record)
                # 构建返回结果
                training_result = {
                    "status": "success",
                    "message": f"训练完成（模式：{reason}）",
                    "training_stats": {
                        "processed_samples": len(df),
                        "saved_to_db": saved_count,
                        "new_samples": new_samples,
                        "total_samples": self.total_samples,
                        "training_mode": reason,
                        "evaluation_rmse": agg_rmse,
                        "training_duration": round(train_duration, 2),
                        "record_id": record_id,
                        "mining_features_enabled": self.mining_advance_enabled,
                        "mining_features_calculated": 'cumulative_advance' in df.columns,
                        "mining_samples_count": df[
                        'daily_advance'].notnull().sum() if 'daily_advance' in df.columns else 0,
                        # ============ 20251218新增时空特征统计 ============
                        "spatiotemporal_features_enabled": hasattr(self.data_preprocessor,
                                                                   'spatiotemporal_extractor') and \
                                                           self.data_preprocessor.spatiotemporal_extractor is not None,
                        "spatiotemporal_feature_count": len(new_features) if 'new_features' in locals() else 0,
                        "feature_categories": categories if 'categories' in locals() else {}
                        # ============ 20251218新增结束 ============
                    },
                    "evaluation_details": eval_result if eval_result.get("status") == "success" else None
                }
                rmse_str = f"{agg_rmse:.4f}" if agg_rmse is not None else "无评估数据"
                logger.info(f"训练后评估RMSE: {rmse_str}")
                # ============ 只在训练成功完成后添加诊断信息输出 ============
                if training_result.get("status") == "success":
                    # 输出训练诊断信息
                    self._print_training_diagnosis()
                return training_result
            except Exception as e:
                # 训练失败 → 回滚事务
                logger.error(f"训练失败，触发回滚：{str(e)}", exc_info=True)
                # 训练失败时也记录监控
                train_duration = (datetime.now() - train_start).total_seconds()
                from performance_monitor import global_monitor
                global_monitor.record_training_session(
                    train_mode="failed",
                    sample_count=len(df) if 'df' in locals() else 0,
                    duration=train_duration,
                    rmse=None
                )
                if db_trans:
                    try:
                        db_trans.rollback()
                        logger.info("事务已回滚，无残留数据")
                    except Exception as rollback_e:
                        if custom_create_time and db_conn:
                            from sqlalchemy import text
                            delete_sql = text("DELETE FROM t_prediction_parameters WHERE create_time = :ct")
                            db_conn.execute(delete_sql, {"ct": custom_create_time})
                            db_trans.commit()
                            logger.info(f"手动删除残留数据（create_time：{custom_create_time}）")
                return {
                    "status": "error",
                    "message": str(e),
                    "training_stats": {
                        "processed_samples": len(df) if 'df' in locals() else 0,
                        "saved_to_db": saved_count,
                        "data_rolled_back": True
                    }
                }
            finally:
                if db_conn:
                    try:
                        db_conn.close()
                        logger.debug("训练流程数据库连接已关闭")
                    except Exception as close_e:
                        logger.warning(f"关闭连接失败：{str(close_e)}")

    def evaluate_model(self, eval_size=200, eval_df=None, use_fixed_set=False):
        """
        增强的模型评估方法，支持固定评估集
        :param eval_size: 评估样本数
        :param eval_df: 外部评估数据
        :param use_fixed_set: 是否使用固定评估集
        :return: 评估结果
        """
        if use_fixed_set and hasattr(self, 'fixed_evaluation_set') and self.fixed_evaluation_set is not None:
            logger.info("使用固定评估集进行评估")
            eval_df = self.fixed_evaluation_set
        return self.model_evaluator.evaluate_model(
            self.models, self.preprocessor, self.training_features, self.data_preprocessor.target_features,
            self._fitted_feature_order, self.db, eval_size, eval_df
        )

    def predict(self, data):
        """
        模型预测（委托给ModelPredictor）
        新增：添加进尺特征状态
        """
        # 使用ModelPredictor进行预测
        result = self.model_predictor.predict(
            data, self.models, self.preprocessor, self.training_features,
            self.data_preprocessor.target_features, self._fitted_feature_order, self.is_trained,
            self.file_lock, self.data_preprocessor, self.fault_calculator, self.db
        )

        # ============ 新增：添加进尺特征状态 ============
        if hasattr(self, 'mining_advance_enabled'):
            # 在结果中添加进尺特征状态
            if 'success' in result and result['success']:
                result['mining_features'] = {
                    'enabled': self.mining_advance_enabled,
                    'message': '进尺特征已启用' if self.mining_advance_enabled else '进尺特征未启用'
                }
        # ============ 新增结束 ============
        return result

    def rollback_model(self, backup_index=-1):
        """模型回滚（委托给ModelManager）"""
        result = self.model_manager.rollback_model(backup_index, self.data_preprocessor.target_features)
        if result["success"]:
            # 重新加载模型
            self._load_model()
            # 同步状态到预处理器
            self.data_preprocessor.is_trained = self.is_trained
            self.data_preprocessor.training_features = self.training_features
        return result
    def get_model_status(self):
        """获取模型当前状态"""
        backup_count = 0
        backup_root = os.path.join(self.model_dir, "backup")
        if os.path.exists(backup_root):
            try:
                backup_count = len(os.listdir(backup_root))
            except Exception:
                backup_count = 0
        latest_eval = self.eval_history[-1] if self.eval_history else None
        return {
            "is_trained": self.is_trained,
            "total_samples": self.total_samples,
            "training_features_count": len(self.training_features) if self.training_features else 0,
            "target_features": self.data_preprocessor.target_features,
            "backup_count": backup_count,
            "latest_evaluation": latest_eval,
            "last_train_time": self.training_stats[-1]["timestamp"] if self.training_stats else None
        }


    def create_fixed_evaluation_set(self, data, size=50):
        """
        创建固定的评估数据集
        确保不同训练阶段使用相同的评估基准

        :param data: 训练数据
        :param size: 评估集大小
        :return: 固定评估数据集
        """
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            # 确保数据量足够
            if len(df) < size:
                logger.warning(f"数据量不足({len(df)}条)，无法创建{size}条的固定评估集")
                return df
            # 按工作面分层采样，确保评估集代表性
            fixed_eval_set = []
            if 'workface_id' in df.columns:
                workface_groups = df.groupby('workface_id')
                for workface_id, group in workface_groups:
                    group_size = max(1, int(size * len(group) / len(df)))
                    if len(group) >= group_size:
                        sampled = group.sample(n=group_size, random_state=42)
                        fixed_eval_set.append(sampled)
                if fixed_eval_set:
                    fixed_eval_df = pd.concat(fixed_eval_set, ignore_index=True)
                    # 如果总数超过size，随机采样调整
                    if len(fixed_eval_df) > size:
                        fixed_eval_df = fixed_eval_df.sample(n=size, random_state=42)
                else:
                    fixed_eval_df = df.sample(n=size, random_state=42)
            else:
                fixed_eval_df = df.sample(n=size, random_state=42)

            logger.info(f"创建固定评估集: {len(fixed_eval_df)}条数据")
            return fixed_eval_df

        except Exception as e:
            logger.error(f"创建固定评估集失败: {str(e)}")
            # 失败时回退到随机采样
            return df.sample(n=min(size, len(df)), random_state=42)

    def set_fixed_evaluation_set(self, data, size=50):
        """
        设置固定评估数据集供后续使用
        """
        self.fixed_evaluation_set = self.create_fixed_evaluation_set(data, size)
        logger.info(f"固定评估集已设置: {len(self.fixed_evaluation_set)}条数据")
        return self.fixed_evaluation_set

    def retrain_from_db_full(self, workface_id=None, sample_limit=None, force_full_train=True):
        """
        全量重新训练方法（防止模型被误删除，强制全量训练）

        :param workface_id: int，可选，筛选特定工作面数据
        :param sample_limit: int，可选，限制训练样本数（避免内存溢出）
        :param force_full_train: bool，是否强制全量训练（默认True）
        :return: dict，训练结果
        """
        with self.file_lock:
            self._print_header("全量重新训练模型（从数据库恢复）")
            retrain_start = datetime.now()
            try:
                # Step 1: 从数据库读取历史数据（使用模型管理器）
                logger.info(f"从数据库读取历史数据：工作面对{workface_id}，样本限制{sample_limit}")
                df = self.model_manager.get_recent_data_from_db(self.db, limit=sample_limit)
                if 'measurement_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['measurement_date']):
                    df['measurement_date'] = df['measurement_date'].astype('int64') // 10 ** 9  # 转为 Unix 时间戳
                    logger.info("数据库读取的 measurement_date 已转换为 Unix 时间戳")
                if df.empty:
                    msg = "未从数据库读取到任何数据，无法重新训练"
                    logger.warning(msg)
                    self._print_result(msg)
                    return {
                        "status": "warning",
                        "message": msg,
                        "training_stats": {"processed_samples": 0, "training_performed": False}
                    }
                # Step 2: 筛选特定工作面数据
                if workface_id is not None and 'workface_id' in df.columns:
                    original_count = len(df)
                    df = df[df["workface_id"] == workface_id].reset_index(drop=True)
                    if df.empty:
                        msg = f"工作面ID={workface_id} 无数据"
                        logger.warning(msg)
                        return {"status": "warning", "message": msg}
                    logger.info(f"筛选工作面ID={workface_id}，样本数：{original_count} → {len(df)}")
                # Step 3: 数据预处理（重新计算断层影响系数）
                logger.info("全量重新训练：自动计算断层影响系数")
                try:
                    df = self.calculate_fault_influence_strength(df)
                except Exception as e:
                    logger.error(f"计算断层影响系数失败：{str(e)}，使用现有值")
                    # 即使失败也继续，使用现有值
                # Step 4: 数据预处理（训练模式）
                try:
                    df_processed, training_features = self.data_preprocessor.preprocess_data(
                        df, is_training=True, fault_calculator=self.fault_calculator, db_utils=self.db
                    )
                except Exception as e:
                    logger.error(f"数据预处理失败：{str(e)}")
                    raise ValueError(f"数据预处理失败：{str(e)}")
                # Step 5: 检查最小样本数
                if len(df_processed) < self.min_train_samples:
                    msg = f"样本数 {len(df_processed)} < 最小训练样本数 {self.min_train_samples}，无法重新训练"
                    logger.warning(msg)
                    self._print_result(msg)
                    return {
                        "status": "warning",
                        "message": msg,
                        "training_stats": {"processed_samples": len(df_processed), "training_performed": False}
                    }
                # Step 6: 强制配置为全量训练模式
                logger.info("强制使用全量训练配置")
                # 切换到全量训练配置（phase1）
                config_before = self.current_config
                if self.current_config != "config_phase1.ini":
                    logger.info(f"切换到全量训练配置：{config_before} → config_phase1.ini")
                    self.reload_config("config_phase1.ini", reload_database=False)
                # Step 7: 构建训练集
                X = df_processed[training_features]
                y = df_processed[self.data_preprocessor.target_features].values
                fitted_feature_order = X.columns.tolist()
                # Step 8: 特征预处理
                preprocessor, X_proc, _ = self.model_trainer.create_preprocessor(
                    X, self.data_preprocessor.base_categorical
                )
                # Step 9: 执行全量训练（强制使用全量训练逻辑）
                logger.info(
                    f"开始全量训练，样本数：{len(df_processed)}，目标数：{len(self.data_preprocessor.target_features)}")
                models = self.model_trainer._full_train(X_proc, y, self.data_preprocessor.target_features)
                # Step 10: 更新模型状态
                self.models = models
                self.preprocessor = preprocessor
                self.training_features = training_features
                self._fitted_feature_order = fitted_feature_order
                self.is_trained = True
                self.data_preprocessor.is_trained = True
                self.data_preprocessor.training_features = training_features

                # Step 11: 保存模型（不触发备份，因为是恢复训练）
                logger.info("保存重新训练的模型")
                self.model_manager.save_model(self.models, self.preprocessor, self.training_features,
                                              self.data_preprocessor.target_features)
                # Step 12: 模型评估
                eval_result = None
                if len(df_processed) > 5:  # 避免小数据评估不准确
                    try:
                        eval_result = self.evaluate_model(eval_size=min(50, len(df_processed)), eval_df=df_processed)
                        logger.info(
                            f"重新训练后评估结果：状态={eval_result.get('status')}, RMSE={eval_result.get('avg_rmse')}")
                    except Exception as e:
                        logger.warning(f"重新训练后评估失败：{str(e)}")
                # Step 13: 恢复原配置（如果需要）
                if config_before != "config_phase1.ini":
                    logger.info(f"恢复原配置：config_phase1.ini → {config_before}")
                    self.reload_config(config_before, reload_database=False)
                # Step 14: 计算训练耗时
                train_duration = (datetime.now() - retrain_start).total_seconds()
                # Step 15: 记录监控
                from performance_monitor import global_monitor
                global_monitor.record_training_session(
                    train_mode="full_retrain",  # 特殊标记为全量重新训练
                    sample_count=len(df_processed),
                    duration=train_duration,
                    rmse=eval_result.get("avg_rmse") if eval_result else None
                )
                # Step 16: 记录训练历史
                training_record = {
                    "sample_count": len(df_processed),
                    "total_samples": self.total_samples,  # 注意：不更新总样本数
                    "train_mode": "full_retrain",
                    "status": "success",
                    "message": f"全量重新训练完成，样本数：{len(df_processed)}，RMSE：{eval_result.get('avg_rmse') if eval_result else '未评估'}",
                    "duration": train_duration,
                    "train_time": datetime.now()
                }
                record_id = self.db.insert_training_record(training_record)
                # Step 17: 构建返回结果
                training_result = {
                    "status": "success",
                    "message": f"全量重新训练完成（样本数：{len(df_processed)}）",
                    "training_stats": {
                        "processed_samples": len(df_processed),
                        "training_mode": "full_retrain",
                        "evaluation_rmse": eval_result.get("avg_rmse") if eval_result else None,
                        "training_duration": round(train_duration, 2),
                        "record_id": record_id,
                        "workface_filtered": workface_id is not None,
                        "sample_limit_applied": sample_limit is not None
                    },
                    "evaluation_details": eval_result if eval_result and eval_result.get(
                        "status") == "success" else None
                }
                # 输出训练诊断信息
                if training_result.get("status") == "success":
                    self._print_training_diagnosis()
                logger.info(f"全量重新训练成功完成，耗时：{train_duration:.2f}秒，样本数：{len(df_processed)}")
                return training_result
            except Exception as e:
                # 训练失败处理
                train_duration = (datetime.now() - retrain_start).total_seconds()
                logger.error(f"全量重新训练失败：{str(e)}", exc_info=True)

                # 记录监控
                from performance_monitor import global_monitor
                global_monitor.record_training_session(
                    train_mode="full_retrain_failed",
                    sample_count=len(df) if 'df' in locals() else 0,
                    duration=train_duration,
                    rmse=None
                )
                return {
                    "status": "error",
                    "message": str(e),
                    "training_stats": {
                        "processed_samples": len(df) if 'df' in locals() else 0,
                        "training_duration": round(train_duration, 2),
                        "training_performed": False
                    }
                }