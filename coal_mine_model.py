"""
煤矿瓦斯风险预测系统 - 主模型类
整合所有模块，提供统一的模型接口
"""
import os
from datetime import datetime

import numpy as np
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
        # 算法锁定：一旦存在已训练模型（algorithm.pkl），系统运行中将忽略配置文件里的algorithm变更
        self.locked_algorithm = None
        self.total_samples = 0
        self.eval_history = []
        self.training_stats = []
        self.baseline_rmse = None
        self._fitted_feature_order = None

        # 注意：不再包含瓦斯涌出量相关状态
        self.note = "模型已重构，瓦斯涌出量请使用独立的/calculate_gas_emission_source接口计算"
        # Step 6: 初始化数据库工具与跨进程锁
        self.db = DBUtils(config_path=config_path)
        self.file_lock = FileLock(self.model_manager.lock_file_path)
        logger.info(f"跨进程锁初始化完成，锁文件路径：{self.model_manager.lock_file_path}")
        # Step 7: 加载已有模型与同步数据库样本数
        self._load_model()
        # Step 7.1: 算法锁定（若已有模型，后续reload_config/retrain不得切换算法）
        locked = self.model_manager.get_locked_algorithm()
        if locked:
            self.locked_algorithm = locked
            # 覆盖trainer算法（即使配置文件被修改）
            if getattr(self.model_trainer, "algorithm", None) != locked:
                logger.warning(
                    f"检测到已训练模型算法锁定为 {locked}，将覆盖当前配置/Trainer算法（启动后不可切换）"
                )
            self.model_trainer.algorithm = locked
        else:
            # 尚未训练过模型：以当前配置为准，但记录为“预期锁定算法”
            self.locked_algorithm = getattr(self.model_trainer, "algorithm", None)

        try:
            self.total_samples = self.model_manager.get_total_samples_from_db(self.db)
        except Exception as e:
            self.total_samples = 0
            logger.warning(f"同步数据库样本数失败：{str(e)}，初始化为0")

        # Step 8: 控制台输出初始化结果
        self._print_header("模型初始化完成")
        self.current_config = config_path  # 记录当前使用的配置文件
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
            current_locked_algorithm = getattr(self, 'locked_algorithm', None)
            current_modules = {
                'fault_calculator': self.fault_calculator,
                'regional_calculator': self.regional_calculator,
                'data_preprocessor': self.data_preprocessor,
                'model_trainer': self.model_trainer,
                'model_evaluator': self.model_evaluator,
                'model_manager': self.model_manager,
                'db': self.db
            }
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

            # Step 3.1: 算法锁定（存在已训练模型时，忽略配置文件中的algorithm变更）
            locked = current_locked_algorithm or self.model_manager.get_locked_algorithm()
            if locked:
            # 读取用户配置中的algorithm（可能不存在）
                cfg_algo = merged_config.get("Model", "algorithm", fallback=locked).strip().lower()
                if cfg_algo != locked:
                   logger.warning(
                        f"配置文件algorithm={cfg_algo}将被忽略；系统已锁定算法={locked}（启动后不可切换）"
                   )
                # 强制写回合并后的配置，确保后续重建Trainer使用锁定算法
                if not merged_config.has_section("Model"):
                    merged_config.add_section("Model")
                merged_config.set("Model", "algorithm", locked)
                self.locked_algorithm = locked
            else:
                # 未训练模型时，允许以配置为准，并记录为预期锁定算法
                self.locked_algorithm = merged_config.get("Model", "algorithm",fallback="lightgbm").strip().lower()

            # Step 4: 重新初始化配置工具和各模块,使用合并后的配置对象来初始化各模块
            self.config_utils = ConfigUtils(self.config_path)
            # 重新初始化各功能模块
            self.fault_calculator = FaultCalculator(self.config_path)
            self.regional_calculator = RegionalMeasureCalculator(self.config_path)
            self.data_preprocessor = DataPreprocessor(self.config_path)
            self.model_trainer = ModelTrainer(self.config)
            self.model_evaluator = ModelEvaluator(self.config_path)
            self.model_manager = ModelManager(self.model_dir)
            # ------------------------------------------------------------------
            # 增量训练窗口参数（用于单天批次自动补历史窗口，避免时序漂移）
            # ------------------------------------------------------------------
            try:
                self.incremental_lookback_days = self.config.getint(
                    "Model", "incremental_lookback_days", fallback=7
                )
            except Exception:
                self.incremental_lookback_days = 7

            try:
                self.incremental_window_limit = self.config.getint(
                    "Model", "incremental_window_limit", fallback=3000
                )
            except Exception:
                self.incremental_window_limit = 3000

            logger.info(
                f"增量训练窗口参数加载完成："
                f"lookback_days={self.incremental_lookback_days}, "
                f"window_limit={self.incremental_window_limit}"
            )

            # Step 5: 条件性重载数据库配置
            if reload_database:
                logger.info("重载数据库配置（将重建数据库连接）")
                db_reload_success = self.db.reload_config(self.config_path)
                if not db_reload_success:
                    logger.error("数据库配置重载失败，但继续其他配置重载")
            else:
                logger.info("跳过数据库配置重载（使用现有连接）")
            # Step 6: 重新加载模型核心参数
            self.full_train_threshold = self.config_utils._get_config_value("Model", "full_train_threshold", 6,
                                                                            is_int=True)
            self.min_train_samples = self.config_utils._get_config_value("Model", "min_train_samples", 6,
                                                                         is_int=True)
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
            # 恢复算法锁定状态
            self.locked_algorithm = current_locked_algorithm
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
                # Step 1: 数据预处理 - 添加详细日志和错误处理
                logger.info(f"开始数据预处理，输入数据样本数: {len(data) if isinstance(data, list) else 'unknown'}")

                try:
                    df = self._preprocess_data(data, is_training=True)
                    logger.info(f"数据预处理完成，DataFrame形状: {df.shape}")
                except Exception as e:
                    logger.error(f"数据预处理失败: {str(e)}", exc_info=True)
                    raise ValueError(f"数据预处理失败: {str(e)}")

                # 检查关键列是否存在
                logger.info(f"DataFrame列名: {list(df.columns)}")
                logger.info(f"训练特征: {self.training_features}")
                logger.info(f"目标特征: {self.data_preprocessor.target_features}")

                # ============ 先执行：自动配置切换逻辑（确保训练特征以最终配置为准） ============
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

                # 配置切换后再打印一次，便于审计最终生效的特征配置
                logger.info(f"[配置切换后] 训练特征: {self.training_features}")
                # ============ 强制剔除 gas_emission_q 相关特征（根治缺列导致训练失败） ============
                try:
                    if self.training_features:
                        before = list(self.training_features)
                        self.training_features = [
                            f for f in self.training_features
                            if not (f == "gas_emission_q"
                                    or f.startswith("gas_emission_q_")
                                    or "gas_emission_q_" in f)
                        ]
                        removed = [f for f in before if f not in self.training_features]
                        if removed:
                            logger.warning(f"已强制剔除 {len(removed)} 个 gas_emission_q 相关特征：{removed}")
                            logger.info(f"剔除后训练特征数：{len(before)} -> {len(self.training_features)}")
                            logger.info(f"剔除后训练特征列表：{self.training_features}")

                        # 同步给 data_preprocessor（保证训练/预测/评估一致）
                        if hasattr(self.data_preprocessor, "training_features"):
                            self.data_preprocessor.training_features = self.training_features
                except Exception as _e:
                    logger.warning(f"强制剔除 gas_emission_q 特征失败（已忽略）：{repr(_e)}", exc_info=True)
                # ============ 强制剔除结束 ============

                # ============ 新增：自动特征降级 / 自动恢复（升级） ============
                # 冷启动阶段常见：days_* / distance_time_interaction / advance_rate 恒为0或无信息量，自动剔除降噪；
                # 当后续数据具备时间跨度/推进信息时自动恢复。
                try:
                    if not self.training_features:
                        raise ValueError("training_features为空，无法执行自动特征降级")

                    # 固化“配置层面”的全量特征（首次进入train时记录一次）
                    if not hasattr(self, "_configured_training_features") or not self._configured_training_features:
                        self._configured_training_features = list(self.training_features)

                    degrade_candidates = [
                        "days_since_start",
                        "days_in_workface",
                        "distance_time_interaction",
                        "advance_rate",
                    ]

                    def _is_degenerate_feature(_df: pd.DataFrame, col: str):
                        """判断特征是否退化；返回(是否退化, 原因)"""
                        if col not in _df.columns:
                            return True, "缺列"
                        s = _df[col]
                        # 统一处理inf
                        try:
                            if pd.api.types.is_numeric_dtype(s):
                                s = s.replace([np.inf, -np.inf], np.nan)
                        except Exception:
                            pass
                        # 全空
                        if s.isna().all():
                            return True, "全缺失"
                        # 常数列（含全为0）
                        try:
                            nunq = int(s.nunique(dropna=True))
                        except Exception:
                            nunq = 2  # 保守：不判退化
                        if nunq <= 1:
                            if pd.api.types.is_numeric_dtype(s):
                                try:
                                    if (s.fillna(0) == 0).all():
                                        return True, "常数列（全为0）"
                                except Exception:
                                    pass
                            return True, "常数列（nunique<=1）"
                        return False, "OK"
                    # 1) 自动降级：剔除退化特征
                    removed = []
                    reasons_map = {}
                    for c in degrade_candidates:
                        deg, why = _is_degenerate_feature(df, c)
                        if deg and c in self.training_features:
                            removed.append(c)
                            reasons_map[c] = why
                    # 2) 自动恢复：当退化特征在新数据中有信息量则恢复（按配置顺序）
                    restored = []
                    for c in degrade_candidates:
                        if c in getattr(self, "_configured_training_features", []) and c not in self.training_features:
                            deg, _ = _is_degenerate_feature(df, c)
                            if not deg:
                                restored.append(c)
                    if removed or restored:
                        before_cnt = len(self.training_features)
                        active = list(self.training_features)
                        # 先恢复（按配置顺序）
                        if restored:
                            cfg = list(self._configured_training_features)
                            active_set = set(active) | set(restored)
                            active = [x for x in cfg if x in active_set]
                        # 再剔除
                        if removed:
                            active = [x for x in active if x not in set(removed)]
                        self.training_features = active
                        # 同步到预处理器（预测/评估对齐需要）
                        if hasattr(self.data_preprocessor, "training_features"):
                            self.data_preprocessor.training_features = self.training_features
                        after_cnt = len(self.training_features)
                        if removed:
                            logger.warning(
                                f"自动特征降级：剔除{len(removed)}个退化特征 -> {removed}，原因={reasons_map}"
                            )
                        if restored:
                            logger.info(f"自动特征恢复：恢复{len(restored)}个特征 -> {restored}")
                        logger.info(f"本次训练生效特征数：{before_cnt} -> {after_cnt}")
                        logger.info(f"本次训练生效特征列表：{self.training_features}")
                except Exception as _e:
                    logger.warning(f"自动特征降级/恢复执行失败（已忽略）：{repr(_e)}", exc_info=True)
                # ============ 自动特征降级 / 自动恢复结束 ============

                if len(df) < self.min_train_samples:
                    msg = f"样本数 {len(df)} < 最小训练样本数 {self.min_train_samples}，跳过训练"
                    logger.warning(msg)
                    self._print_result(msg)
                    return {
                        "status": "warning",
                        "message": msg,
                        "training_stats": {"processed_samples": len(df), "training_performed": False}
                    }
                # 因为增量训练会回捞lookback窗口，最终训练集(train_df)可能通过DB补齐增强特征列
                missing_features = []
                if self.training_features:
                    missing_features = [f for f in self.training_features if f not in df.columns]
                if missing_features:
                    logger.warning(f"本批训练df缺少特征(可能由窗口train_df补齐)：{missing_features}")
                    logger.debug(f"本批df可用列: {list(df.columns)}")
                missing_targets = []
                if self.data_preprocessor.target_features:
                    missing_targets = [t for t in self.data_preprocessor.target_features if t not in df.columns]
                if missing_targets:
                    logger.error(f"训练数据缺少目标特征: {missing_targets}")
                    raise ValueError(f"训练数据缺少目标特征: {missing_targets}")
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
                # Step 4: 判断训练模式（先判断模式，再决定是否允许重建预处理器）
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
                # Step 5: 构建训练集（增量训练：只要进入增量模式就强制lookback窗口，确保必然生效）
                train_df = df
                if (not do_full_train) and self.is_trained:
                    # 读取窗口参数
                    try:
                        lookback_days = self.config.getint("Model", "incremental_lookback_days", fallback=14)
                    except Exception:
                        lookback_days = 14
                    try:
                        window_limit = self.config.getint("Model", "incremental_window_limit", fallback=2000)
                    except Exception:
                        window_limit = 2000
                    # 以“本批次最大日期”作为窗口上界（保证同事务未提交数据也能纳入）
                    max_date = None
                    try:
                        if "measurement_date" in df.columns:
                            _mx = pd.to_datetime(df["measurement_date"], errors="coerce").max()
                            if pd.notna(_mx):
                                max_date = str(_mx.date())
                    except Exception:
                        max_date = None
                    workface_ids = None
                    try:
                        if "workface_id" in df.columns:
                            workface_ids = sorted({int(float(x)) for x in df["workface_id"].dropna().unique()})
                    except Exception:
                        workface_ids = None
                    try:
                        # 说明：
                        #  - 这里用同一个 db_conn（同事务）取数，可“看到”刚插入未提交的数据
                        #  - 从而保证窗口一定包含当前批次 + 历史lookback
                        train_df = self.db.fetch_recent_training_window_with_features(
                            workface_ids=workface_ids,
                            max_date=max_date,
                            lookback_days=int(lookback_days),
                            limit=int(window_limit),
                            conn=db_conn
                        )
                        if train_df is None or train_df.empty:
                            logger.warning("增量训练窗口取数为空，回退为仅本批次df训练")
                            train_df = df
                        else:
                            logger.info(
                                f"增量训练窗口已生效：lookback_days={lookback_days}，window_limit={window_limit}，"
                                f"实际训练样本={len(train_df)}（含当前批次）"
                            )
                    except Exception as _e:
                        logger.warning(f"增量训练窗口拉取失败（回退为仅本批次df训练）：{repr(_e)}", exc_info=True)
                        train_df = df
                # 训练目标 y 必须来自 train_df（否则窗口即使拼了也等于没生效）
                y = train_df[self.data_preprocessor.target_features].values
                if (not do_full_train) and self.is_trained:
                    # --- 增量训练：强制使用已fit的特征顺序与已fit的预处理器 ---
                    if not getattr(self, "_fitted_feature_order", None):
                        logger.warning("增量训练：未找到已fit特征顺序，自动切换为全量训练")
                        do_full_train = True
                        reason = "特征顺序缺失触发全量训练"
                    if self.preprocessor is None:
                        logger.warning("增量训练：未找到已fit预处理器，自动切换为全量训练")
                        do_full_train = True
                        reason = "预处理器缺失触发全量训练"

                if do_full_train:
                    # 全量训练：允许使用当前training_features并重新fit预处理器
                    X = train_df[self.training_features]
                    self._fitted_feature_order = X.columns.tolist()
                    logger.info(f"构建训练集: X形状={X.shape}, y形状={y.shape}")
                    self.preprocessor, X_proc, _ = self.model_trainer.create_preprocessor(
                        X, self.data_preprocessor.base_categorical
                    )
                else:
                    # 增量训练：固定特征集合，缺失列补0，避免“自动特征降级/恢复”导致维度变化
                    fitted_cols = list(self._fitted_feature_order)
                    missing = [c for c in fitted_cols if c not in train_df.columns]
                    if missing:
                        # 这里不直接报错，而是补0：因为有些增强特征在新批次/历史窗口可能暂时不可得
                        logger.warning(f"增量训练：训练集(df, 含窗口)缺失{len(missing)}个已fit特征，将补0：{missing}")
                    X = train_df.reindex(columns=fitted_cols, fill_value=0)
                    logger.info(f"构建训练集(增量): X形状={X.shape}, y形状={y.shape}")
                    # 关键：只transform，不再fit，确保输出维度恒定
                    X_proc = self.preprocessor.transform(X)
                # Step 6: 执行训练
                if do_full_train:
                    self.models = self.model_trainer._full_train(X_proc, y, self.data_preprocessor.target_features)
                else:
                    self.models = self.model_trainer._incremental_train(
                        X_proc, y, self.data_preprocessor.target_features, self.models
                    )
                # 防御：训练器若异常返回非dict（历史遗留），立即失败触发回滚，避免“数据提交但模型异常”
                if not isinstance(self.models, dict):
                    raise ValueError(f"训练失败：models类型异常({type(self.models).__name__})，将触发回滚")

                # Step 8: 提交事务+更新状态
                db_trans.commit()
                logger.info(f"事务提交成功，{saved_count} 条数据已持久化")
                # 关键修复：事务已提交，后续即使评估失败也不允许再rollback
                db_trans = None
                self.total_samples = self.model_manager.get_total_samples_from_db(self.db)
                new_samples = self.total_samples - initial_samples
                self._print_step(f"样本数更新：新增 {new_samples} 条，累计 {self.total_samples} 条")
                # Step 9: 保存模型+标记训练状态
                self.model_manager.save_model(
                    self.models, self.preprocessor, self.training_features, self.data_preprocessor.target_features
                )
                self.is_trained = True
                self.data_preprocessor.is_trained = True
                self.data_preprocessor.training_features = self.training_features
                # Step 10: 训练后评估（评估失败只降级，不影响训练成功）
                if len(df) <= 5:
                    eval_result = {
                        "status": "skipped",
                        "message": f"数据量过少({len(df)}条)，跳过评估避免误判",
                        "avg_rmse": None
                    }
                    agg_rmse = None
                    logger.info(f"小数据训练({len(df)}条)：跳过性能评估")
                else:
                    try:
                        eval_result = self.evaluate_model()
                        agg_rmse = eval_result.get("avg_rmse")
                    except Exception as _e:
                        # 评估失败不应导致训练失败，更不允许回滚已提交数据
                        logger.error(f"训练后评估失败（将降级为warning，不影响训练提交）：{repr(_e)}", exc_info=True)
                        eval_result = {"status": "warning", "message": f"评估失败：{str(_e)}"}
                        agg_rmse = None
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
                        "record_id": record_id
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
                # 关键：str(e) 可能被外层 KeyError 等覆盖，增加 repr(e) 便于定位真实根因
                logger.error(f"训练失败，触发回滚（str）：{str(e)}", exc_info=True)
                logger.error(f"训练失败，触发回滚（repr）：{repr(e)}", exc_info=True)
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
                    # 返回 message 同时带上 repr(e)，避免只看到 "'pred_id'" 这种被覆盖的信息
                    "message": f"{str(e)} | {repr(e)}",
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
            self._fitted_feature_order, self.db, eval_size, eval_df,
            data_preprocessor=self.data_preprocessor
        )

    def predict(self, data):
        """模型预测（委托给ModelPredictor）"""
        return self.model_predictor.predict(
            data, self.models, self.preprocessor, self.training_features,
            self.data_preprocessor.target_features, self._fitted_feature_order, self.is_trained,
            self.file_lock, self.data_preprocessor, self.fault_calculator, self.db
        )

    def retrain_from_db(self, workface_id=None, limit=None):
        """从数据库重新训练模型（向后兼容）"""
        self._print_header("从数据库重新训练模型")

        # 记录向后兼容的警告
        logger.warning("retrain_from_db方法已过时，请使用retrain_from_db_full方法")

        try:
            # 从数据库读取历史数据
            df = self.model_manager.get_recent_data_from_db(self.db, limit=limit)
            if df.empty:
                msg = "未从数据库读取到任何数据，无法重新训练"
                logger.warning(msg)
                self._print_result(msg)
                return {"status": "warning", "message": msg}

            # 筛选特定工作面数据
            if workface_id is not None and 'workface_id' in df.columns:
                df = df[df["workface_id"] == workface_id].reset_index(drop=True)
                self._print_step(f"筛选工作面ID={workface_id}，剩余样本数：{len(df)}")
                if df.empty:
                    msg = f"工作面ID={workface_id} 无数据"
                    logger.warning(msg)
                    return {"status": "warning", "message": msg}

            # 重新计算断层系数
            logger.info("重新训练：自动计算断层影响系数")
            df = self.calculate_fault_influence_strength(df)

            # 执行训练（使用新的全量重新训练方法）
            result = self.retrain_from_db_full(
                workface_id=workface_id,
                sample_limit=limit,
                force_full_train=True
            )
            return result
        except Exception as e:
            logger.error(f"重新训练失败：{str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

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

    def _save_training_stats(self, train_mode, sample_count, agg_rmse):
        """保存训练统计到内存"""
        if not hasattr(self, 'training_stats'):
            self.training_stats = []
        self.training_stats.append({
            "timestamp": datetime.now(),
            "train_mode": train_mode,
            "sample_count": sample_count,
            "total_samples": self.total_samples,
            "agg_rmse": agg_rmse
        })
        self.training_stats = self.training_stats[-100:]
        logger.debug(f"训练统计更新，累计 {len(self.training_stats)} 条记录")

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