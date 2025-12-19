"""
煤矿瓦斯风险预测系统 - 数据预处理模块
包含：数据预处理、特征工程、分源特征计算
"""
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy import create_engine

from config_utils import ConfigUtils


class DataPreprocessor(ConfigUtils):
    """数据预处理器"""

    def __init__(self, config_path="config.ini"):
        super().__init__(config_path)
        self.fixed_reference_date = None
        self.date_reference = None
        self.target_features = None
        self._load_feature_config()  # 在初始化时加载特征配置

        # ============ 新增：加载进尺配置 ============
        self._load_mining_advance_config()
        self.spatiotemporal_extractor = None
        self._init_spatiotemporal_extractor(config_path)
        # ============ 新增结束 ============

        # 添加缺失的属性
        self.is_trained = False
        self.training_features = None

    def _init_spatiotemporal_extractor(self, config_path):
        """
        初始化时空特征提取器
        """
        try:
            self.spatiotemporal_extractor = SpatiotemporalFeatureExtractor(config_path)
            logger.info("时空特征提取器初始化成功")
        except Exception as e:
            logger.warning(f"初始化时空特征提取器失败: {str(e)}，将不使用时空特征")
            self.spatiotemporal_extractor = None

    def _load_mining_advance_config(self):
        """
        私有方法：加载回采进尺相关配置（[MiningAdvance] section）
        """
        try:
            self.default_daily_advance = self._get_config_value(
                "MiningAdvance", "default_daily_advance", 3.0, is_float=True
            )

            # 使用config.getboolean读取布尔值
            self.enable_cumulative_advance = self.config.getboolean(
                "MiningAdvance", "enable_cumulative_advance", fallback=True
            )
            self.enable_effective_exposure = self.config.getboolean(
                "MiningAdvance", "enable_effective_exposure", fallback=True
            )

            self.advance_data_gap_threshold = self._get_config_value(
                "MiningAdvance", "advance_data_gap_threshold", 3, is_int=True
            )

            logger.info(
                f"回采进尺配置加载完成：默认日进尺={self.default_daily_advance}m，"
                f"累计进尺特征={'启用' if self.enable_cumulative_advance else '禁用'}，"
                f"有效暴露距离特征={'启用' if self.enable_effective_exposure else '禁用'}"
            )

        except Exception as e:
            logger.warning(f"加载回采进尺配置失败：{str(e)}，使用默认值")
            self._set_default_mining_advance_params()

    def _set_default_mining_advance_params(self):
        """私有方法：回采进尺默认参数（配置缺失时兜底）"""
        self.default_daily_advance = 3.0  # 默认每日进尺3米
        self.enable_cumulative_advance = True
        self.enable_effective_exposure = True
        self.advance_data_gap_threshold = 3
        logger.debug("回采进尺已设置默认参数")


    def preprocess_data(self, data, is_training=True, fault_calculator=None, db_utils=None):
        """
        公开方法：数据预处理（统一格式、清洗、特征生成，兼容训练/预测模式）

        :param data: list[dict] / pandas.DataFrame，输入数据
        :param is_training: bool，是否为训练模式（True=训练，False=预测/评估）
        :param fault_calculator: FaultCalculator实例，用于断层计算
        :param db_utils: DBUtils实例，用于数据库操作
        :return: pandas.DataFrame，预处理后的数据
        :raises ValueError: 关键特征缺失时抛出
        """
        logger.debug(f"数据预处理开始（训练模式: {'是' if is_training else '否'}），原始样本: {len(data)}")

        # Step 1: 统一数据格式为DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data)
        logger.debug(f"数据格式统一为DataFrame，初始样本数：{len(df)}")
        # ============ 进尺特征处理（在时间特征之前） ============
        if self.enable_cumulative_advance or self.enable_effective_exposure:
            df = self._process_mining_advance_features(df, is_training, db_utils)
        # 时间特征处理（在早期处理以确保后续可用）
        if self.enable_temporal_features:
            df = self._process_temporal_features_simple(df)
        if self.spatiotemporal_extractor:
            try:
                logger.debug("开始提取时空特征")
                df = self.spatiotemporal_extractor.extract_features(df, is_training)
                new_features = self.spatiotemporal_extractor.get_new_feature_names()
                if new_features:
                    logger.info(f"时空特征提取完成，新增 {len(new_features)} 个特征")
                    logger.debug(
                        f"新增特征: {', '.join(new_features[:10])}{'...' if len(new_features) > 10 else ''}")
            except Exception as e:
                logger.error(f"提取时空特征时出错: {str(e)}", exc_info=True)
        else:
            logger.debug("未启用时空特征提取器")

        # Step 2: 自动补充断层影响系数（缺失时计算）
        if 'fault_influence_strength' not in df.columns or df['fault_influence_strength'].isnull().any():
            logger.debug("检测到 fault_influence_strength 缺失，自动计算")
            if fault_calculator and db_utils:
                df_dict = fault_calculator.calculate_fault_influence_strength(df.to_dict('records'), db_utils)
                df = pd.DataFrame(df_dict)
            else:
                logger.warning("缺少断层计算器或数据库工具，无法计算断层影响系数")
                # 设置默认值
                df['fault_influence_strength'] = 0.5

        # Step 3: 校验区域措施强度（训练/预测均需提前计算）
        if 'regional_measure_strength' not in df.columns or df['regional_measure_strength'].isnull().any():
            raise ValueError(
                "数据缺少 regional_measure_strength！需先调用 /api/model/calculate_regional_strength 接口计算"
            )

        # Step 4: 列名标准化（处理空格/横杠，避免字段匹配错误）
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.replace(' ', '_')
            .str.replace('-', '_')
        )
        logger.debug(f"列名标准化完成，当前列：{df.columns.tolist()}")

        # 修正目标列名映射（确保与标准化后的DataFrame一致）
        normalized_cols = df.columns.tolist()
        fixed_targets = []
        for t in self.target_features:
            for c in normalized_cols:
                if c.lower() == t.lower():
                    fixed_targets.append(c)
                    break
        if set(fixed_targets) != set(self.target_features):
            logger.warning(f"目标列名自动修正：{self.target_features} → {fixed_targets}")
        self.target_features = fixed_targets

        # Step 5: 数据去重（基于位置标识，保留最新记录）
        identifier_cols = ['working_face', 'roadway', 'roadway_id', 'distance_from_entrance']
        available_identifiers = [col for col in identifier_cols if col in df.columns]
        if available_identifiers:
            dup_cnt = df.duplicated(subset=available_identifiers).sum()
            if dup_cnt > 0:
                df = df.drop_duplicates(subset=available_identifiers, keep='last').reset_index(drop=True)
                logger.debug(f"数据去重：移除 {dup_cnt} 条重复记录，剩余 {len(df)} 条")
        else:
            logger.warning("位置标识列（working_face/roadway等）不全，无法完整去重")

        # Step 6: 按掘进距离排序（确保时序逻辑，distance_from_entrance 升序）
        if 'distance_from_entrance' in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df['distance_from_entrance']):
                    df = df.sort_values(by='distance_from_entrance', ascending=True).reset_index(drop=True)
                    logger.debug("按 distance_from_entrance 升序排序完成")
                else:
                    logger.warning("distance_from_entrance 非数值类型，跳过排序")
            except Exception as e:
                logger.warning(f"排序失败：{str(e)}")

        # ============ 修改：缺失值填充部分，处理Timestamp类型 ============
        # Step 7: 缺失值填充（分类用众数，数值用中位数，避免异常值干扰）
        # 7.1 分类特征填充（如working_face）
        for col in self.base_categorical:
            if col in df.columns and df[col].isnull().any():
                fill_val = df[col].mode()[0] if not df[col].mode().empty else "未知"
                df[col] = df[col].fillna(fill_val)
                logger.debug(f"分类特征 {col} 缺失值填充：{fill_val}")

        # 7.2 数值特征填充（如coal_thickness）
        for col in self.base_numeric:
            if col in df.columns and df[col].isnull().any():
                # 特殊处理时间相关字段
                if col in ['depth_from_face']:
                    # 验证孔深度缺失时使用默认值0（表示工作面位置）
                    fill_val = 0.0
                    logger.debug(f"验证孔深度 {col} 缺失值填充：{fill_val}")
                elif col == 'measurement_date':
                    # 测量日期缺失时，使用数据中最常见的日期或当前日期
                    # 确保不保留Timestamp对象
                    if not df[col].mode().empty:
                        fill_val = df[col].mode()[0]
                        # 如果是Timestamp，转换为datetime
                        if isinstance(fill_val, pd.Timestamp):
                            fill_val = fill_val.to_pydatetime()
                    else:
                        # 使用当前日期
                        fill_val = datetime.now()
                    logger.debug(f"测量日期 {col} 缺失值填充：{fill_val}")
                else:
                    # 检查列的数据类型
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_val = df[col].median() if not df[col].isna().all() else 0.0
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        # 如果是日期类型，使用最常见的日期
                        if not df[col].mode().empty:
                            fill_val = df[col].mode()[0]
                        else:
                            fill_val = datetime.now()
                    else:
                        # 其他类型使用0.0
                        fill_val = 0.0
                    logger.debug(f"数值特征 {col} 缺失值填充：{fill_val}")

                df[col] = df[col].fillna(fill_val)
        # 7.3 目标特征填充（训练/评估时确保无NaN）
        if is_training:
            for col in self.target_features:
                if col in df.columns and df[col].isnull().any():
                    fill_val = df[col].median() if not df[col].isna().all() else 0.0
                    df[col] = df[col].fillna(fill_val)
                    logger.debug(f"目标特征 {col} 缺失值填充：{fill_val}")

        # Step 8: 生成分源特征（AQ1018—2006标准，全参数存在且无缺失才计算）
        source_prediction_params = [
            "coal_thickness", "tunneling_speed", "initial_gas_emission_strength", "roadway_length",
            "roadway_cross_section", "coal_density", "original_gas_content", "residual_gas_content"
        ]

        all_params_exist = all(param in df.columns for param in source_prediction_params)
        if all_params_exist:
            if df[source_prediction_params].isnull().any().any():
                all_params_exist = False
                logger.warning("分源参数存在缺失值，不计算分源特征")

        if all_params_exist:
            df["gas_emission_wall"] = df.apply(
                lambda r: self._calculate_coal_wall_emission(
                    r["coal_thickness"], r["tunneling_speed"],
                    r["initial_gas_emission_strength"], r["roadway_length"]
                ), axis=1
            )
            df["gas_emission_fallen"] = df.apply(
                lambda r: self._calculate_fallen_coal_emission(
                    r["roadway_cross_section"], r["coal_density"],
                    r["tunneling_speed"], r["original_gas_content"], r["residual_gas_content"]
                ), axis=1
            )
            df["total_gas_emission"] = df["gas_emission_wall"] + df["gas_emission_fallen"]
            logger.debug(
                f"分源特征计算完成：平均总瓦斯涌出量 {df['total_gas_emission'].mean():.4f} m³/min"
            )
        else:
            df["gas_emission_wall"] = 0.0
            df["gas_emission_fallen"] = 0.0
            df["total_gas_emission"] = 0.0
            logger.debug("分源参数不全，分源特征设为0.0")

        # Step 9: 确保所有期望特征存在（缺失则填充默认值）
        all_features = self.base_categorical + self.base_numeric + [
            'gas_emission_wall', 'gas_emission_fallen', 'total_gas_emission'
        ]
        # ============ 20251218新增：包含时空特征 ============
        if self.spatiotemporal_extractor:
            new_features = self.spatiotemporal_extractor.get_new_feature_names()
            if new_features:
                all_features.extend(new_features)
                logger.debug(f"特征列表扩展，包含 {len(new_features)} 个时空特征")
        # ============ 20251218新增结束 ============
        for col in all_features:
            if col not in df.columns:
                fill_val = "未知" if col in self.base_categorical else 0.0
                df[col] = fill_val
                logger.debug(f"特征 {col} 缺失，填充默认值：{fill_val}")

        # Step 10: 训练/预测模式差异化处理
        if is_training:
            missing_targets = [t for t in self.target_features if t not in df.columns]
            if missing_targets:
                raise ValueError(f"训练数据缺少目标特征：{missing_targets}")
            training_features = all_features
            logger.debug(f"训练特征确定：共 {len(training_features)} 个")
            return df, training_features
        else:
            if not self.training_features:
                raise ValueError("模型未训练，无法确定预测特征顺序")
            keep_cols = self.training_features + self.target_features
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols]
            logger.debug(f"预测数据对齐：按训练特征顺序保留 {len(df.columns)} 个字段")
        if is_training:
            self._log_data_quality_summary(df)
            return df
        return df

    def _load_feature_config(self):
        """
        私有方法：加载特征配置（[Features] section）
        简化版：移除推进距离相关配置，专注于测量日期和深度
        """
        try:
            # Step 1: 读取分类特征（如工作面、巷道）
            categorical_str = self.config.get("Features", "base_categorical", fallback="")
            self.base_categorical = [x.strip() for x in categorical_str.split(",") if x.strip()]

            # Step 2: 读取数值特征（如坐标、煤层厚度、测量日期、验证孔深度）
            numeric_str = self.config.get("Features", "base_numeric", fallback="")
            self.base_numeric = [x.strip() for x in numeric_str.split(",") if x.strip()]

            # Step 3: 读取预测目标特征（如瓦斯涌出量Q）
            target_str = self.config.get("Features", "target_features", fallback="")
            self.target_features = [x.strip() for x in target_str.split(",") if x.strip()]

            # Step 4: 读取时间特征配置（从[TemporalFeatures] section，简化版）
            try:
                # 检查是否启用时间特征
                self.enable_temporal_features = self.config.getboolean(
                    "TemporalFeatures", "enable_temporal_features", fallback=True
                )

                # 读取日期参考基准配置
                self.date_reference = self.config.get(
                    "TemporalFeatures", "date_reference", fallback="min_date"
                )

                # 读取固定参考日期（如果配置了）
                fixed_reference_str = self.config.get(
                    "TemporalFeatures", "fixed_reference_date", fallback=""
                )
                if fixed_reference_str:
                    from datetime import datetime
                    try:
                        self.fixed_reference_date = datetime.strptime(fixed_reference_str, "%Y-%m-%d")
                        logger.debug(f"固定参考日期配置成功：{self.fixed_reference_date}")
                    except ValueError as ve:
                        logger.warning(f"固定参考日期格式错误：{fixed_reference_str}，错误：{str(ve)}")
                        self.fixed_reference_date = None
                else:
                    self.fixed_reference_date = None

            except Exception as temporal_e:
                # 时间特征配置加载失败时使用默认值
                logger.warning(f"加载时间特征配置失败：{str(temporal_e)}，使用默认值")
                self.enable_temporal_features = True
                self.date_reference = "min_date"
                self.fixed_reference_date = None

            # Step 5: 验证时间相关特征是否在base_numeric中（优化检查）
            # 注意：days_since_workface_start 和 advance_distance 是动态计算的，不需要在配置中
            temporal_base_fields = ['measurement_date', 'depth_from_face']

            if self.enable_temporal_features:
                # 只检查基础字段是否在配置中
                missing_temporal_in_config = [
                    field for field in temporal_base_fields
                    if field not in self.base_numeric
                ]

                if missing_temporal_in_config:
                    logger.warning(
                        f"启用了时间特征，但以下基础时间字段不在base_numeric配置中：{missing_temporal_in_config}。"
                        f"系统将在数据预处理时自动处理这些字段。"
                    )

                logger.info(f"时间特征处理已启用，日期参考基准：{self.date_reference}")
            else:
                logger.info("时间特征处理已禁用，将仅使用空间和工程特征")

            # Step 6: 校验特征配置有效性（保持原有逻辑）
            if not self.base_categorical:
                logger.warning("未配置基础分类特征（base_categorical），可能影响模型精度")

            if not self.base_numeric:
                logger.warning("未配置基础数值特征（base_numeric），模型无法训练")

            if not self.target_features:
                raise ValueError("必须配置至少一个预测目标特征（target_features）")

            # Step 7: 记录加载的特征统计信息
            temporal_fields_in_config = [field for field in temporal_base_fields if field in self.base_numeric]

            logger.info(
                f"特征配置加载完成：\n"
                f"  - 分类特征：{len(self.base_categorical)}个 ({', '.join(self.base_categorical[:3])}{'...' if len(self.base_categorical) > 3 else ''})\n"
                f"  - 数值特征：{len(self.base_numeric)}个 ({', '.join(self.base_numeric[:3])}{'...' if len(self.base_numeric) > 3 else ''})\n"
                f"  - 目标特征：{len(self.target_features)}个 ({', '.join(self.target_features)})\n"
                f"  - 时间特征：{'已启用' if self.enable_temporal_features else '已禁用'} "
                f"(基础字段：{len(temporal_fields_in_config)}/{len(temporal_base_fields)})"
            )

        except Exception as e:
            logger.error(f"加载特征配置失败：{str(e)}", exc_info=True)

            # 失败时设置默认值以确保系统继续运行
            self.base_categorical = []
            self.base_numeric = []
            self.target_features = []
            self.enable_temporal_features = False
            self.date_reference = "min_date"
            self.fixed_reference_date = None

            logger.warning("特征配置加载失败，已设置为空值，模型可能无法训练")

            # 重新抛出异常，避免后续训练出现更严重错误
            raise ValueError(f"特征配置加载失败：{str(e)}")

    def _log_data_quality_summary(self, df):
        """简单的数据质量摘要日志"""
        try:
            # 基础检查
            if hasattr(self, 'target_features'):
                for target in self.target_features:
                    if target in df.columns:
                        variance = df[target].var()
                        if variance < 0.1:
                            logger.warning(f"🚨 目标特征 {target} 方差过低: {variance:.6f}")

            # 缺失值检查
            missing_columns = df.columns[df.isnull().any()].tolist()
            if missing_columns:
                logger.warning(f"⚠️ 数据包含缺失值的列: {missing_columns}")

        except Exception as e:
            logger.debug(f"数据质量检查失败: {str(e)}")

    def _calculate_coal_wall_emission(self, coal_thickness, tunneling_speed, initial_strength, roadway_length):
        """私有方法：计算煤壁瓦斯涌出量（AQ1018—2006公式）"""
        try:
            if tunneling_speed <= 0:
                logger.warning("掘进速度≤0，煤壁涌出量设为0")
                return 0.0
            roadway_length = max(roadway_length, 0.0)
            val = coal_thickness * tunneling_speed * initial_strength * (
                    2 * np.sqrt(roadway_length / (tunneling_speed + 1e-9)) - 1
            )
            return max(0.0, float(val))
        except Exception as e:
            logger.error(f"计算煤壁涌出量失败：{str(e)}", exc_info=True)
            return 0.0

    def _calculate_fallen_coal_emission(self, cross_section, coal_density, tunneling_speed, original_gas, residual_gas):
        """私有方法：计算落煤瓦斯涌出量（AQ1018—2006公式）"""
        try:
            gas_diff = max(0.0, (original_gas or 0.0) - (residual_gas or 0.0))
            val = cross_section * coal_density * tunneling_speed * gas_diff
            return max(0.0, float(val))
        except Exception as e:
            logger.error(f"计算落煤涌出量失败：{str(e)}", exc_info=True)
            return 0.0

    # ============ 修改：_process_temporal_features_simple方法中的日期处理 ============
    def _process_temporal_features_simple(self, df):
        """
        私有方法：简化版时间特征处理
        基于测量日期、坐标和验证孔深度唯一确定测量状态

        处理逻辑：
        1. 转换测量日期为datetime格式
        2. 创建时间数值特征（避免类别特征）
        3. 创建时空唯一标识

        :param df: pandas.DataFrame，输入数据
        :return: pandas.DataFrame，添加时间特征后的数据
        """
        try:
            logger.debug("开始处理时间特征（简化版）")

            # 1. 确保measurement_date列存在并转换为datetime
            if 'measurement_date' in df.columns:
                # 转换日期字符串为datetime
                # 首先检查是否已经是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(df['measurement_date']):
                    df['measurement_date'] = pd.to_datetime(
                        df['measurement_date'],
                        errors='coerce',  # 转换失败设为NaT
                        format='%Y-%m-%d'  # 支持YYYY-MM-DD格式
                    )
                else:
                    # 如果已经是datetime，确保时区一致
                    df['measurement_date'] = df['measurement_date'].dt.tz_localize(None)

                # 2. 检查日期有效性
                invalid_dates = df['measurement_date'].isna().sum()
                if invalid_dates > 0:
                    logger.warning(f"发现 {invalid_dates} 条记录的测量日期格式无效")
                    # 填充无效日期为最早有效日期
                    if not df['measurement_date'].isna().all():
                        min_valid_date = df['measurement_date'].min()
                        df['measurement_date'] = df['measurement_date'].fillna(min_valid_date)
                    else:
                        # 如果所有日期都无效，使用当前日期
                        df['measurement_date'] = pd.Timestamp.now().normalize()

            else:
                logger.warning("数据中缺少measurement_date字段，跳过时间特征处理")
                return df

            # 3. 创建时间数值特征
            df = self._create_temporal_numeric_features(df)

            # 4. 创建时空唯一标识（用于追踪同一位置不同时间的测量）
            df = self._create_spatiotemporal_id(df)

            logger.debug("简化版时间特征处理完成")
            return df

        except Exception as e:
            logger.error(f"处理时间特征失败：{str(e)}，跳过时间特征处理", exc_info=True)
            return df

    # ============ 20251218新增：进尺特征处理方法 ============
    def _process_mining_advance_features(self, df, is_training=True, db_utils=None):
        """
        私有方法：处理回采进尺相关特征
        核心功能：计算累计进尺和有效暴露距离

        :param df: pandas.DataFrame，输入数据
        :param is_training: 是否为训练模式
        :param db_utils: 数据库工具实例，用于预测时查询历史数据
        :return: pandas.DataFrame，添加进尺特征后的数据
        """
        try:
            logger.debug(f"开始处理回采进尺特征（模式：{'训练' if is_training else '预测'}）")

            # Step 1: 确保daily_advance字段存在
            if 'daily_advance' not in df.columns:
                logger.warning("数据中缺少daily_advance字段，使用默认值")
                df['daily_advance'] = self.default_daily_advance
            else:
                # 填充缺失的日进尺数据
                daily_advance_missing = df['daily_advance'].isnull().sum()
                if daily_advance_missing > 0:
                    logger.warning(f"daily_advance有{daily_advance_missing}条缺失值，使用默认值填充")
                    df['daily_advance'] = df['daily_advance'].fillna(self.default_daily_advance)

            # Step 2: 确保measurement_date字段存在且为datetime格式
            if 'measurement_date' not in df.columns:
                logger.warning("数据中缺少measurement_date字段，无法准确计算累计进尺")
                # 如果没有日期，使用简单累加
                df['cumulative_advance'] = df['daily_advance'].cumsum().round(2)
            else:
                # 确保measurement_date为datetime格式
                if not pd.api.types.is_datetime64_any_dtype(df['measurement_date']):
                    df['measurement_date'] = pd.to_datetime(
                        df['measurement_date'], errors='coerce', format='%Y-%m-%d'
                    )

                # Step 3: 按工作面计算累计进尺
                if 'workface_id' in df.columns:
                    # 按工作面分组计算
                    df['cumulative_advance'] = 0.0

                    for workface_id, group in df.groupby('workface_id'):
                        # 按测量日期排序
                        group_sorted = group.sort_values('measurement_date')

                        # 训练模式：从0开始累计
                        # 预测模式：尝试从数据库获取历史累计进尺
                        if is_training:
                            cumulative_advance = 0.0
                        else:
                            # 预测模式：尝试获取该工作面上次的累计进尺
                            cumulative_advance = self._get_last_cumulative_advance(workface_id, db_utils)

                        prev_date = None

                        for idx, row in group_sorted.iterrows():
                            current_date = row['measurement_date']

                            # 检查数据是否连续（计算与上次测量的天数差）
                            if prev_date is not None and pd.notnull(current_date) and pd.notnull(prev_date):
                                days_diff = (current_date - prev_date).days

                                # 如果数据中断超过阈值，重新开始累计
                                if days_diff > self.advance_data_gap_threshold:
                                    if is_training:
                                        logger.debug(f"工作面{workface_id}数据中断{days_diff}天，重置累计进尺")
                                    else:
                                        logger.warning(f"工作面{workface_id}数据中断{days_diff}天，累计进尺可能不连续")
                                    cumulative_advance = 0.0

                            # 累加当日进尺
                            daily_advance = row['daily_advance']
                            if pd.notnull(daily_advance):
                                cumulative_advance += daily_advance

                            # 更新累计进尺值
                            df.loc[idx, 'cumulative_advance'] = round(cumulative_advance, 2)
                            prev_date = current_date

                    logger.debug(f"按工作面计算累计进尺完成，唯一工作面数：{df['workface_id'].nunique()}")
                else:
                    # 如果没有工作面ID，按日期排序计算全局累计进尺
                    logger.warning("数据中缺少workface_id字段，按全局计算累计进尺")
                    df = df.sort_values('measurement_date')
                    df['cumulative_advance'] = df['daily_advance'].cumsum().round(2)

            # Step 4: 计算有效暴露距离（如果启用）
            if self.enable_effective_exposure:
                if 'depth_from_face' in df.columns and 'cumulative_advance' in df.columns:
                    df['effective_exposure_distance'] = (
                            df['cumulative_advance'] + df['depth_from_face']
                    ).round(2)

                    logger.debug(
                        f"有效暴露距离计算完成，范围：[{df['effective_exposure_distance'].min():.1f}, "
                        f"{df['effective_exposure_distance'].max():.1f}]米"
                    )
                else:
                    logger.warning("缺少depth_from_face或cumulative_advance字段，无法计算有效暴露距离")
                    df['effective_exposure_distance'] = 0.0

            # Step 5: 创建进尺相关衍生特征
            df = self._create_mining_advance_derived_features(df)

            logger.debug("回采进尺特征处理完成")
            return df

        except Exception as e:
            logger.error(f"处理回采进尺特征失败：{str(e)}，跳过此步骤", exc_info=True)
            # 失败时添加默认值，确保流程继续
            df['cumulative_advance'] = df.get('daily_advance', self.default_daily_advance)
            df['effective_exposure_distance'] = df.get('depth_from_face', 0.0)
            return df

    def _get_last_cumulative_advance(self, workface_id, db_utils):
        """
        获取工作面上次的累计进尺（用于预测模式）

        :param workface_id: 工作面ID
        :param db_utils: 数据库工具实例
        :return: 上次累计进尺值，如果没有则返回0
        """
        if db_utils is None:
            logger.warning("未提供数据库工具，无法查询历史累计进尺")
            return 0.0

        try:
            # 从数据库查询该工作面上次训练时的累计进尺
            # 注意：这里假设数据库中已保存了训练时的累计进尺
            db_conf = db_utils.db_config
            db_url = f"mysql+pymysql://{db_conf['user']}:{db_conf['password']}@" \
                     f"{db_conf['host']}:{db_conf['port']}/{db_conf['db']}?charset={db_conf['charset']}"
            engine = create_engine(db_url)

            with engine.connect() as conn:
                from sqlalchemy import text
                # 查询该工作面的最大累计进尺
                query = text("""
                    SELECT MAX(cumulative_advance) as last_advance
                    FROM t_prediction_parameters 
                    WHERE workface_id = :workface_id 
                      AND cumulative_advance IS NOT NULL
                """)

                result = conn.execute(query, {"workface_id": workface_id})
                row = result.fetchone()

                if row and row[0] is not None:
                    last_advance = float(row[0])
                    logger.debug(f"查询到工作面{workface_id}的历史累计进尺：{last_advance}")
                    return last_advance
                else:
                    logger.warning(f"工作面{workface_id}无历史累计进尺记录")
                    return 0.0

        except Exception as e:
            logger.error(f"查询历史累计进尺失败：{str(e)}")
            return 0.0

    def _create_mining_advance_derived_features(self, df):
        """
        私有方法：创建进尺相关衍生特征

        :param df: pandas.DataFrame，输入数据
        :return: pandas.DataFrame，添加衍生特征后的数据
        """
        try:
            # 1. 进尺变化率（当日进尺/累计进尺，避免除以零）
            if 'cumulative_advance' in df.columns and 'daily_advance' in df.columns:
                df['advance_change_rate'] = df.apply(
                    lambda row: (
                        row['daily_advance'] / row['cumulative_advance']
                        if row['cumulative_advance'] > 0
                        else 0.0
                    ),
                    axis=1
                ).round(4)

            # 2. 进尺阶段分组（每20米一个阶段）
            if 'cumulative_advance' in df.columns:
                # 创建进尺阶段
                max_advance = df['cumulative_advance'].max()
                num_stages = max(1, int(max_advance / 20) + 1)
                bins = [i * 20 for i in range(num_stages + 1)]
                labels = [f"阶段{i + 1}({bins[i]}-{bins[i + 1]}米)" for i in range(num_stages)]

                df['advance_stage'] = pd.cut(
                    df['cumulative_advance'],
                    bins=bins,
                    labels=labels[:num_stages],
                    right=False
                )

                logger.debug(f"进尺阶段分组完成，共{num_stages}个阶段")

            # 3. 进尺速度类别
            if 'daily_advance' in df.columns:
                # 使用分位数确定速度类别阈值
                q33 = df['daily_advance'].quantile(0.33) if len(df) > 0 else 2.0
                q66 = df['daily_advance'].quantile(0.66) if len(df) > 0 else 4.0

                def classify_speed(val):
                    if val <= q33:
                        return "慢速"
                    elif val <= q66:
                        return "中速"
                    else:
                        return "快速"

                df['advance_speed_category'] = df['daily_advance'].apply(classify_speed)

                # 统计各速度类别数量
                speed_counts = df['advance_speed_category'].value_counts()
                logger.debug(f"进尺速度分类完成：{dict(speed_counts)}")

            return df

        except Exception as e:
            logger.warning(f"创建进尺衍生特征失败：{str(e)}，跳过此步骤")
            return df

    # ============ 20251218 新增结束 ============

    def _create_temporal_numeric_features(self, df):
        """
        私有方法：创建时间数值特征

        :param df: pandas.DataFrame，输入数据
        :return: pandas.DataFrame，添加时间数值特征后的数据
        """
        try:
            if 'measurement_date' in df.columns and not df['measurement_date'].isna().all():
                # 确保measurement_date是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(df['measurement_date']):
                    df['measurement_date'] = pd.to_datetime(df['measurement_date'], errors='coerce')

                # 确定参考日期
                if self.date_reference == "fixed_date" and self.fixed_reference_date:
                    reference_date = self.fixed_reference_date
                else:
                    # 使用最早测量日期作为参考
                    reference_date = df['measurement_date'].min()
                    logger.debug(f"使用最早测量日期作为参考日期: {reference_date}")

                # 确保reference_date是datetime
                if isinstance(reference_date, pd.Timestamp):
                    reference_date = reference_date.to_pydatetime()

                # 计算从参考日期开始的天数
                df['days_since_reference'] = (df['measurement_date'] - reference_date).dt.days

                # 确保天数为非负（对于参考日期之前的日期）
                df['days_since_reference'] = df['days_since_reference'].clip(lower=0)

                # 提取日期组件（可选） - 转换为数值类型
                df['day_of_week'] = df['measurement_date'].dt.dayofweek.astype(float)  # 周几（0=周一）
                df['day_of_month'] = df['measurement_date'].dt.day.astype(float)  # 月中的第几天
                df['month'] = df['measurement_date'].dt.month.astype(float)  # 月份

                # 确保没有NaN值
                date_features = ['days_since_reference', 'day_of_week', 'day_of_month', 'month']
                for feat in date_features:
                    if feat in df.columns:
                        df[feat] = df[feat].fillna(0).astype(float)

                logger.debug(
                    f"创建时间数值特征完成：参考日期={reference_date}，日期范围={df['measurement_date'].min()}到{df['measurement_date'].max()}"
                )

            return df
        except Exception as e:
            logger.warning(f"创建时间数值特征失败：{str(e)}，跳过此步骤")
            return df

    def _create_spatiotemporal_id(self, df):
        """
        私有方法：创建时空唯一标识
        格式：坐标_日期_深度，用于唯一标识一个测量状态

        :param df: pandas.DataFrame，输入数据
        :return: pandas.DataFrame，添加时空标识后的数据
        """
        try:
            # 检查必需的字段
            required_fields = ['x_coord', 'y_coord', 'z_coord', 'measurement_date', 'depth_from_face']
            missing_fields = [field for field in required_fields if field not in df.columns]

            if missing_fields:
                logger.warning(f"无法创建时空标识，缺少字段：{missing_fields}")
                return df

            # 创建时空唯一标识
            df['spatiotemporal_id'] = (
                    df['x_coord'].round(1).astype(str) + '_' +
                    df['y_coord'].round(1).astype(str) + '_' +
                    df['z_coord'].round(1).astype(str) + '_' +
                    df['measurement_date'].dt.strftime('%Y%m%d') + '_' +
                    df['depth_from_face'].round(1).astype(str)
            )

            # 检查是否有重复的时空标识（不应该有）
            duplicate_count = df['spatiotemporal_id'].duplicated().sum()
            if duplicate_count > 0:
                logger.warning(f"发现 {duplicate_count} 个重复的时空标识，可能存在重复数据")

            logger.debug(f"创建时空标识完成，唯一标识数量：{df['spatiotemporal_id'].nunique()}")
            return df
        except Exception as e:
            logger.warning(f"创建时空标识失败：{str(e)}，跳过此步骤")
            return df


# ============ 20251218 新增：时空特征提取器类 ============
class SpatiotemporalFeatureExtractor:
    """
    时空特征提取器
    负责从原始数据中提取时空相关特征，增强模型对时空模式的学习能力
    """

    def __init__(self, config_path="config.ini"):
        """
        初始化时空特征提取器
        :param config_path: 配置文件路径
        """
        from loguru import logger
        self.logger = logger

        # 加载配置
        import configparser
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding="utf-8")

        # 时空特征相关参数
        self.spatial_radius = self.config.getfloat("SpatialFeatures", "spatial_radius", fallback=15.0)
        self.temporal_lag_days = self.config.getint("TemporalFeatures", "temporal_lag_days", fallback=5)
        self.enable_spatial_features = self.config.getboolean("SpatialFeatures", "enable_spatial_features",
                                                              fallback=True)
        self.enable_temporal_features = self.config.getboolean("TemporalFeatures", "enable_temporal_features",
                                                               fallback=True)

        self.logger.info(
            f"时空特征提取器初始化完成: 空间半径={self.spatial_radius}m, 时间滞后={self.temporal_lag_days}天")

    def extract_features(self, df, is_training=True):
        """
        提取时空特征
        :param df: pandas.DataFrame，预处理后的数据
        :param is_training: bool，是否为训练模式
        :return: pandas.DataFrame，包含新特征的数据
        """
        try:
            self.logger.debug("开始提取时空特征")

            # 创建数据副本
            df_enhanced = df.copy()

            # 1. 提取空间特征
            if self.enable_spatial_features:
                df_enhanced = self._extract_spatial_features(df_enhanced)

            # 2. 提取时间序列特征
            if self.enable_temporal_features:
                df_enhanced = self._extract_temporal_features(df_enhanced)

            # 3. 提取衰减规律特征
            df_enhanced = self._extract_decay_features(df_enhanced)

            # 4. 提取交互特征
            df_enhanced = self._extract_interaction_features(df_enhanced)

            self.logger.debug(f"时空特征提取完成，新增特征数: {len(df_enhanced.columns) - len(df.columns)}")

            # 记录新增的特征名
            original_columns = set(df.columns)
            enhanced_columns = set(df_enhanced.columns)
            new_features = list(enhanced_columns - original_columns)

            # 存储新增特征名，供外部访问
            self.new_feature_names = new_features

            return df_enhanced

        except Exception as e:
            self.logger.error(f"提取时空特征失败: {str(e)}", exc_info=True)
            # 失败时返回原始数据
            return df

    def _extract_spatial_features(self, df):
        """
        提取空间特征
        :param df: pandas.DataFrame
        :return: pandas.DataFrame，包含空间特征
        """
        try:
            self.logger.debug("开始提取空间特征")

            # 检查必需字段
            required_cols = ['x_coord', 'y_coord', 'z_coord', 'measurement_date']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"缺少空间特征所需字段: {missing_cols}，跳过空间特征提取")
                return df

            # 1. 创建位置标识（四舍五入到1米精度）
            df['position_key'] = (
                    df['x_coord'].round(1).astype(str) + '_' +
                    df['y_coord'].round(1).astype(str) + '_' +
                    df['z_coord'].round(1).astype(str)
            )

            # 2. 按日期分组，计算空间邻居特征
            from scipy.spatial import KDTree
            import numpy as np

            # 存储空间特征
            spatial_features = []

            for date, group in df.groupby('measurement_date'):
                if len(group) < 2:
                    # 单个点无法计算空间邻居
                    continue

                # 提取坐标
                coords = group[['x_coord', 'y_coord', 'z_coord']].values

                # 构建KD树进行快速空间搜索
                tree = KDTree(coords)

                # 为每个点查找邻居
                for i, (idx, row) in enumerate(group.iterrows()):
                    # 查找半径内的邻居（不包括自己）
                    neighbors = tree.query_ball_point(coords[i], self.spatial_radius)
                    neighbors = [n for n in neighbors if n != i]  # 排除自身

                    if neighbors:
                        neighbor_data = group.iloc[neighbors]

                        # 计算邻居统计特征
                        row_data = row.copy()

                        # 邻居瓦斯参数统计
                        if 'gas_emission_q' in neighbor_data.columns:
                            row_data['neighbor_q_mean'] = neighbor_data['gas_emission_q'].mean()
                            row_data['neighbor_q_std'] = neighbor_data['gas_emission_q'].std()
                            row_data['neighbor_q_min'] = neighbor_data['gas_emission_q'].min()
                            row_data['neighbor_q_max'] = neighbor_data['gas_emission_q'].max()

                        # 邻居钻屑量统计
                        if 'drilling_cuttings_s' in neighbor_data.columns:
                            row_data['neighbor_s_mean'] = neighbor_data['drilling_cuttings_s'].mean()
                            row_data['neighbor_s_std'] = neighbor_data['drilling_cuttings_s'].std()

                        # 邻居深度统计
                        if 'depth_from_face' in neighbor_data.columns:
                            row_data['neighbor_depth_mean'] = neighbor_data['depth_from_face'].mean()

                        # 邻居数量
                        row_data['neighbor_count'] = len(neighbors)

                        # 最近邻距离
                        if len(neighbors) > 0:
                            distances = np.linalg.norm(coords[i] - coords[neighbors], axis=1)
                            row_data['nearest_neighbor_distance'] = distances.min()
                            row_data['mean_neighbor_distance'] = distances.mean()

                        spatial_features.append(row_data)
                    else:
                        # 没有邻居的情况
                        row_data = row.copy()
                        row_data['neighbor_count'] = 0
                        row_data['nearest_neighbor_distance'] = np.nan
                        spatial_features.append(row_data)

            if spatial_features:
                # 合并空间特征
                spatial_df = pd.DataFrame(spatial_features)

                # 将空间特征合并回原数据
                spatial_cols = [col for col in spatial_df.columns if
                                col.startswith('neighbor_') or col in ['nearest_neighbor_distance',
                                                                       'mean_neighbor_distance']]
                spatial_cols.append('position_key')

                for col in spatial_cols:
                    if col in spatial_df.columns:
                        df[col] = spatial_df[col]

            self.logger.debug(
                f"空间特征提取完成，新增 {len([c for c in df.columns if c.startswith('neighbor_')])} 个特征")

            return df

        except Exception as e:
            self.logger.error(f"提取空间特征失败: {str(e)}", exc_info=True)
            return df

    def _extract_temporal_features(self, df):
        """
        提取时间序列特征
        :param df: pandas.DataFrame
        :return: pandas.DataFrame，包含时间特征
        """
        global temporal_cols
        try:
            self.logger.debug("开始提取时间序列特征")

            # 检查必需字段
            if 'position_key' not in df.columns:
                self.logger.warning("缺少position_key字段，无法提取时间序列特征")
                return df

            if 'measurement_date' not in df.columns:
                self.logger.warning("缺少measurement_date字段，无法提取时间序列特征")
                return df

            # 确保日期格式
            if not pd.api.types.is_datetime64_any_dtype(df['measurement_date']):
                df['measurement_date'] = pd.to_datetime(df['measurement_date'], errors='coerce')

            # 按位置分组
            temporal_features = []

            for pos_key, group in df.groupby('position_key'):
                if len(group) < 2:
                    # 单个时间点无法计算时间序列特征
                    continue

                # 按时间排序
                group_sorted = group.sort_values('measurement_date').copy()

                # 计算时间序列特征
                for i, (idx, row) in enumerate(group_sorted.iterrows()):
                    row_data = row.copy()

                    # 1. 测量顺序
                    row_data['measurement_order'] = i + 1

                    # 2. 距离首次测量的天数
                    if i == 0:
                        row_data['days_since_first_measure'] = 0
                    else:
                        first_date = group_sorted.iloc[0]['measurement_date']
                        row_data['days_since_first_measure'] = (row['measurement_date'] - first_date).days

                    # 3. 距离上次测量的天数
                    if i > 0:
                        prev_date = group_sorted.iloc[i - 1]['measurement_date']
                        row_data['days_since_last_measure'] = (row['measurement_date'] - prev_date).days
                    else:
                        row_data['days_since_last_measure'] = np.nan

                    # 4. 累计测量次数
                    row_data['cumulative_measure_count'] = i + 1

                    # 5. 计算变化率（如果有前一次测量）
                    if i > 0:
                        if 'gas_emission_q' in group_sorted.columns:
                            prev_q = group_sorted.iloc[i - 1]['gas_emission_q']
                            if prev_q != 0:
                                row_data['q_change_rate'] = (row['gas_emission_q'] - prev_q) / prev_q
                            else:
                                row_data['q_change_rate'] = np.nan

                        if 'drilling_cuttings_s' in group_sorted.columns:
                            prev_s = group_sorted.iloc[i - 1]['drilling_cuttings_s']
                            if prev_s != 0:
                                row_data['s_change_rate'] = (row['drilling_cuttings_s'] - prev_s) / prev_s
                            else:
                                row_data['s_change_rate'] = np.nan

                    temporal_features.append(row_data)

            if temporal_features:
                # 合并时间特征
                temporal_df = pd.DataFrame(temporal_features)

                # 将时间特征合并回原数据
                temporal_cols = ['measurement_order', 'days_since_first_measure',
                                 'days_since_last_measure', 'cumulative_measure_count',
                                 'q_change_rate', 's_change_rate']

                for col in temporal_cols:
                    if col in temporal_df.columns:
                        # 合并时注意索引对齐
                        if col in df.columns:
                            df[col] = temporal_df[col].combine_first(df[col])
                        else:
                            df[col] = temporal_df[col]

            self.logger.debug(f"时间序列特征提取完成，新增 {len([c for c in temporal_cols if c in df.columns])} 个特征")

            return df

        except Exception as e:
            self.logger.error(f"提取时间序列特征失败: {str(e)}", exc_info=True)
            return df

    def _extract_decay_features(self, df):
        """
        提取衰减规律特征
        :param df: pandas.DataFrame
        :return: pandas.DataFrame，包含衰减特征
        """
        try:
            self.logger.debug("开始提取衰减规律特征")

            # 检查必需字段
            required_cols = ['depth_from_face']
            if 'cumulative_advance' in df.columns:
                required_cols.append('cumulative_advance')
            if 'days_since_first_measure' in df.columns:
                required_cols.append('days_since_first_measure')

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.debug(f"缺少衰减特征所需字段: {missing_cols}，使用简化衰减模型")

            # 1. 深度衰减因子（假设指数衰减）
            if 'depth_from_face' in df.columns:
                # 深度衰减系数（经验值：每米衰减5%）
                depth_decay_coef = 0.05
                df['depth_decay_factor'] = np.exp(-depth_decay_coef * df['depth_from_face'])

            # 2. 时间衰减因子（如果有时间信息）
            if 'days_since_first_measure' in df.columns:
                # 时间衰减系数（经验值：每天衰减1%）
                time_decay_coef = 0.01
                df['time_decay_factor'] = np.exp(-time_decay_coef * df['days_since_first_measure'])

            # 3. 综合衰减因子
            decay_cols = [c for c in df.columns if 'decay_factor' in c]
            if len(decay_cols) > 0:
                df['total_decay_factor'] = df[decay_cols].product(axis=1)

                # 4. 计算衰减调整后的瓦斯值
                if 'gas_emission_q' in df.columns:
                    df['q_decay_adjusted'] = df['gas_emission_q'] / df['total_decay_factor'].clip(lower=0.001)

                if 'drilling_cuttings_s' in df.columns:
                    df['s_decay_adjusted'] = df['drilling_cuttings_s'] / df['total_decay_factor'].clip(lower=0.001)

            # 5. 深度分组特征
            if 'depth_from_face' in df.columns:
                # 创建深度分组（0-2m, 2-4m, 4-6m, 6-8m, 8-10m, 10m+）
                bins = [0, 2, 4, 6, 8, 10, float('inf')]
                labels = ['深度0-2m', '深度2-4m', '深度4-6m', '深度6-8m', '深度8-10m', '深度10m以上']
                df['depth_group'] = pd.cut(df['depth_from_face'], bins=bins, labels=labels, right=False)

            self.logger.debug(
                f"衰减规律特征提取完成，新增 {len([c for c in df.columns if 'decay' in c or 'depth_group' in c])} 个特征")

            return df

        except Exception as e:
            self.logger.error(f"提取衰减规律特征失败: {str(e)}", exc_info=True)
            return df

    def _extract_interaction_features(self, df):
        """
        提取交互特征（空间×时间交互）
        :param df: pandas.DataFrame
        :return: pandas.DataFrame，包含交互特征
        """
        try:
            self.logger.debug("开始提取交互特征")

            # 1. 空间邻居与时间变化的交互
            if 'neighbor_q_mean' in df.columns and 'q_change_rate' in df.columns:
                df['neighbor_q_vs_change'] = df['neighbor_q_mean'] * (1 + df['q_change_rate'].fillna(0))

            # 2. 深度与时间的交互
            if 'depth_from_face' in df.columns and 'days_since_first_measure' in df.columns:
                df['depth_time_interaction'] = df['depth_from_face'] * df['days_since_first_measure']

            # 3. 累计进尺与深度的交互
            if 'cumulative_advance' in df.columns and 'depth_from_face' in df.columns:
                df['advance_depth_interaction'] = df['cumulative_advance'] * df['depth_from_face']

            self.logger.debug(
                f"交互特征提取完成，新增 {len([c for c in df.columns if 'interaction' in c or '_vs_' in c])} 个特征")

            return df

        except Exception as e:
            self.logger.error(f"提取交互特征失败: {str(e)}", exc_info=True)
            return df

    def get_new_feature_names(self):
        """
        获取新增的特征名
        :return: list，新增特征名列表
        """
        return getattr(self, 'new_feature_names', [])

    def get_all_new_feature_categories(self):
        """
        获取所有新增特征的分类
        :return: dict，按类别分组的新增特征
        """
        if not hasattr(self, 'new_feature_names'):
            return {}

        categories = {
            'spatial_features': [],
            'temporal_features': [],
            'decay_features': [],
            'interaction_features': [],
            'other_features': []
        }

        for feature in self.new_feature_names:
            if feature.startswith('neighbor_'):
                categories['spatial_features'].append(feature)
            elif any(keyword in feature for keyword in ['change', 'measure', 'order', 'since']):
                categories['temporal_features'].append(feature)
            elif any(keyword in feature for keyword in ['decay', 'adjusted', 'depth_group']):
                categories['decay_features'].append(feature)
            elif any(keyword in feature for keyword in ['interaction', '_vs_']):
                categories['interaction_features'].append(feature)
            else:
                categories['other_features'].append(feature)

        return categories
# ============ 20251218新增：时空特征提取器类结束 ============