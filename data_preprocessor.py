"""
煤矿瓦斯风险预测系统 - 数据预处理模块
包含：数据预处理、特征工程、分源特征计算
"""
import pandas as pd
import numpy as np
from loguru import logger

from config_utils import ConfigUtils


class DataPreprocessor(ConfigUtils):
    """数据预处理器"""

    def __init__(self, config_path="config.ini"):
        super().__init__(config_path)
        self._load_feature_config()  # 在初始化时加载特征配置

        # 添加缺失的属性
        self.is_trained = False
        self.training_features = None

    def _load_feature_config(self):
        """
        私有方法：加载特征配置（[Features] section）
        注意：已移除瓦斯涌出量相关特征
        """
        try:
            # Step 1: 读取分类特征
            categorical_str = self.config.get("Features", "base_categorical", fallback="")
            self.base_categorical = [x.strip() for x in categorical_str.split(",") if x.strip()]
            # Step 2: 读取数值特征（已移除分源预测法参数）
            numeric_str = self.config.get("Features", "base_numeric", fallback="")
            self.base_numeric = [x.strip() for x in numeric_str.split(",") if x.strip()]
            # Step 3: 读取预测目标特征（只保留钻屑量和瓦斯涌出速度）
            target_str = self.config.get("Features", "target_features", fallback="")
            self.target_features = [x.strip() for x in target_str.split(",") if x.strip()]
            # 校验特征配置有效性
            if not self.base_categorical:
                logger.warning("未配置基础分类特征（base_categorical），可能影响模型精度")
            if not self.base_numeric:
                logger.warning("未配置基础数值特征（base_numeric），模型无法训练")
            if not self.target_features:
                raise ValueError("必须配置至少一个预测目标特征（target_features）")
            logger.debug(
                f"特征配置加载完成（已移除瓦斯涌出量相关特征）："
                f"分类特征：{self.base_categorical}，"
                f"数值特征：{self.base_numeric}，"
                f"目标特征：{self.target_features}"
            )
        except Exception as e:
            logger.error(f"加载特征配置失败：{str(e)}", exc_info=True)
            raise

    def preprocess_data(self, data, is_training=True, fault_calculator=None, db_utils=None):
        """
        公开方法：数据预处理（移除了瓦斯涌出量相关处理）
        """
        logger.debug(f"数据预处理开始（训练模式: {'是' if is_training else '否'}），原始样本: {len(data)}")
        # Step 1: 统一数据格式为DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data)
        # Step 2: 添加时空唯一标识（代替原有去重逻辑）
        df = self._add_spatiotemporal_identifier(df)
        # Step 3: 检查并补充关键时空特征
        df = self._enrich_spatiotemporal_features(df)
        # Step 4: 自动补充断层影响系数
        if 'fault_influence_strength' not in df.columns or df['fault_influence_strength'].isnull().any():
            logger.debug("检测到 fault_influence_strength 缺失，自动计算")
            if fault_calculator and db_utils:
                df_dict = fault_calculator.calculate_fault_influence_strength(df.to_dict('records'), db_utils)
                df = pd.DataFrame(df_dict)
            else:
                df['fault_influence_strength'] = 0.5
        # Step 5: 校验区域措施强度
        if 'regional_measure_strength' not in df.columns or df['regional_measure_strength'].isnull().any():
            raise ValueError(
                "数据缺少 regional_measure_strength！需先调用 /api/model/calculate_regional_strength 接口计算"
            )
        # Step 6: 列名标准化
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.replace(' ', '_')
            .str.replace('-', '_')
        )
        # Step 7: 修正去重逻辑
        original_count = len(df)
        df = df.drop_duplicates(keep='first')
        if len(df) < original_count:
            logger.info(f"移除 {original_count - len(df)} 条完全相同的重复记录")
        # Step 8: 按时间和空间排序
        if 'measurement_date' in df.columns and 'measurement_time' in df.columns:
            try:
                df['measurement_datetime'] = pd.to_datetime(
                    df['measurement_date'] + ' ' + df['measurement_time'].fillna('00:00:00')
                )
                df = df.sort_values(['working_face', 'measurement_datetime',
                                     'distance_from_entrance']).reset_index(drop=True)
                logger.debug("按工作面、测量时间、距入口距离排序")
            except Exception as e:
                logger.warning(f"时间排序失败：{str(e)}")
        elif 'distance_from_entrance' in df.columns:
            df = df.sort_values('distance_from_entrance').reset_index(drop=True)
        # Step 9: 缺失值填充
        # 分类特征填充
        for col in self.base_categorical:
            if col in df.columns and df[col].isnull().any():
                fill_val = df[col].mode()[0] if not df[col].mode().empty else "未知"
                df[col] = df[col].fillna(fill_val)
        # 数值特征填充
        for col in self.base_numeric:
            if col in df.columns and df[col].isnull().any():
                fill_val = df[col].median() if not df[col].isna().all() else 0.0
                df[col] = df[col].fillna(fill_val)
        # 目标特征填充（训练/评估时确保无NaN）
        if is_training:
            for col in self.target_features:
                if col in df.columns and df[col].isnull().any():
                    fill_val = df[col].median() if not df[col].isna().all() else 0.0
                    df[col] = df[col].fillna(fill_val)
        # Step 10: 生成时空特征（新增）
        df = self._generate_spatiotemporal_features(df)
        # Step 11: 确保所有期望特征存在
        # 获取基础特征列表
        base_features = self.base_categorical + self.base_numeric
        # 但我们需要所有实际存在于df中的特征
        all_features = list(df.columns)
        # 移除目标特征和非特征列
        non_feature_cols = self.target_features + ['_spatiotemporal_id', 'measurement_datetime']
        feature_cols = [col for col in all_features if col not in non_feature_cols]
        # 确保所有配置的特征都存在
        for col in base_features:
            if col not in df.columns:
                fill_val = "未知" if col in self.base_categorical else 0.0
                df[col] = fill_val
                logger.debug(f"特征 {col} 缺失，填充默认值：{fill_val}")
                if col not in feature_cols:
                    feature_cols.append(col)
        # Step 12: 训练/预测模式差异化处理
        if is_training:
            missing_targets = [t for t in self.target_features if t not in df.columns]
            if missing_targets:
                raise ValueError(f"训练数据缺少目标特征：{missing_targets}")
            # 确保特征列不包含目标列
            feature_cols = [col for col in feature_cols if col not in self.target_features]
            logger.debug(f"训练特征确定：共 {len(feature_cols)} 个")
            logger.debug(f"特征列: {feature_cols}")
            # 数据质量检查
            self._log_data_quality_summary(df)
            return df, feature_cols
        else:
            if not self.training_features:
                raise ValueError("模型未训练，无法确定预测特征顺序")
            keep_cols = self.training_features
            df = df[keep_cols]
            logger.debug(f"预测数据对齐：按训练特征顺序保留 {len(df.columns)} 个字段")
            return df

    def _add_spatiotemporal_identifier(self, df):
        """
        添加时空唯一标识，避免错误去重
        """
        # 生成复合唯一标识
        identifier_parts = []
        # 基本空间标识
        space_cols = ['working_face', 'x_coord', 'y_coord', 'z_coord']
        for col in space_cols:
            if col in df.columns:
                identifier_parts.append(col)
        # 时间标识（优先）
        if 'measurement_date' in df.columns:
            identifier_parts.append('measurement_date')
        if 'measurement_time' in df.columns:
            identifier_parts.append('measurement_time')
        # 钻孔标识
        if 'borehole_id' in df.columns:
            identifier_parts.append('borehole_id')
        if 'drilling_depth' in df.columns:
            identifier_parts.append('drilling_depth')
        # 距离标识（关键）
        if 'distance_to_face' in df.columns:
            identifier_parts.append('distance_to_face')
        elif 'face_advance_distance' in df.columns:
            identifier_parts.append('face_advance_distance')
        # 生成唯一ID
        if identifier_parts:
            # 检查这些列是否都在df中
            available_parts = [col for col in identifier_parts if col in df.columns]
            if available_parts:
                df['_spatiotemporal_id'] = df[available_parts].astype(str).agg('_'.join, axis=1)
            else:
                df['_spatiotemporal_id'] = df.index.astype(str)
        else:
            df['_spatiotemporal_id'] = df.index.astype(str)
        return df

    def _enrich_spatiotemporal_features(self, df):
        """
        补充关键时空特征
        """
        # 1. 补充距采面距离（如缺失）
        if 'distance_to_face' not in df.columns:
            if 'face_advance_distance' in df.columns and 'drilling_depth' in df.columns:
                # 估算距采面距离 = 钻孔深度 + 工作面推进距离
                df['distance_to_face'] = df['drilling_depth'] + df['face_advance_distance'].fillna(0)
                logger.info("自动计算 distance_to_face 特征")
            else:
                df['distance_to_face'] = 0
                logger.warning("无法计算 distance_to_face，设为0")
        # 2. 创建时间序列特征
        if 'measurement_date' in df.columns:
            try:
                # 转换为时间戳
                df['measurement_date_parsed'] = pd.to_datetime(df['measurement_date'])
                # 计算时间序列特征
                df['days_since_start'] = (df['measurement_date_parsed'] -
                                          df['measurement_date_parsed'].min()).dt.days
                # 按工作面分组的时间序列
                if 'working_face' in df.columns:
                    df['days_in_workface'] = df.groupby('working_face')['measurement_date_parsed'].transform(
                        lambda x: (x - x.min()).dt.days
                    )
                logger.info("时间序列特征生成完成")
            except Exception as e:
                logger.warning(f"时间序列特征生成失败：{str(e)}")
        # 3. 计算相邻测量的变化率（用于检测异常）
        if 'distance_from_entrance' in df.columns and 'working_face' in df.columns:
            # 先按工作面和距离排序
            df = df.sort_values(['working_face', 'distance_from_entrance']).reset_index(drop=True)

            # 计算相邻q、S值的变化
            for target in ['gas_emission_q', 'drilling_cuttings_s', 'gas_emission_velocity_q']:
                if target in df.columns:
                    df[f'{target}_diff'] = df.groupby('working_face')[target].diff()
                    df[f'{target}_pct_change'] = df.groupby('working_face')[target].pct_change()
        return df

    def _generate_spatiotemporal_features(self, df):
        """
        生成时空交互特征
        """
        # 1. 空间-时间交互特征
        if 'distance_to_face' in df.columns and 'days_since_start' in df.columns:
            df['distance_time_interaction'] = df['distance_to_face'] * df['days_since_start'] / 1000
        # 2. 工作面推进特征
        if 'face_advance_distance' in df.columns:
            # 推进速率（如有时间信息）
            if 'measurement_date_parsed' in df.columns and 'working_face' in df.columns:
                # 按工作面分组计算
                advance_rates = []
                for workface, group in df.groupby('working_face'):
                    group_sorted = group.sort_values('measurement_date_parsed')
                    rate = group_sorted['face_advance_distance'].diff() / (
                        group_sorted['measurement_date_parsed'].diff().dt.days.replace(0, 1e-9)
                    )
                    advance_rates.append(rate)
                # 合并结果
                df['advance_rate'] = pd.concat(advance_rates) if advance_rates else 0
        # 3. 历史趋势特征（同一位置的历史q、S值）
        if 'x_coord' in df.columns and 'y_coord' in df.columns and 'z_coord' in df.columns:
            # 创建坐标哈希用于快速匹配
            df['coord_hash'] = (
                    df['x_coord'].round(1).astype(str) + '_' +
                    df['y_coord'].round(1).astype(str) + '_' +
                    df['z_coord'].round(1).astype(str)
            )
            # 计算同一坐标点的历史统计
            for target in ['gas_emission_q', 'drilling_cuttings_s', 'gas_emission_velocity_q']:
                if target in df.columns:
                    # 同一坐标点的历史平均值
                    historical_mean = df.groupby('coord_hash')[target].expanding().mean().reset_index(level=0,
                                                                                                      drop=True)
                    df[f'{target}_historical_mean'] = historical_mean
                    # 同一坐标点的变化趋势
                    df[f'{target}_trend'] = df.groupby('coord_hash')[target].diff()
        return df

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

    # def _calculate_coal_wall_emission(self, coal_thickness, tunneling_speed, initial_strength, roadway_length):
    #     """私有方法：计算煤壁瓦斯涌出量（AQ1018—2006公式）"""
    #     try:
    #         if tunneling_speed <= 0:
    #             logger.warning("掘进速度≤0，煤壁涌出量设为0")
    #             return 0.0
    #         roadway_length = max(roadway_length, 0.0)
    #         val = coal_thickness * tunneling_speed * initial_strength * (
    #                 2 * np.sqrt(roadway_length / (tunneling_speed + 1e-9)) - 1
    #         )
    #         return max(0.0, float(val))
    #     except Exception as e:
    #         logger.error(f"计算煤壁涌出量失败：{str(e)}", exc_info=True)
    #         return 0.0
    #
    # def _calculate_fallen_coal_emission(self, cross_section, coal_density, tunneling_speed, original_gas, residual_gas):
    #     """私有方法：计算落煤瓦斯涌出量（AQ1018—2006公式）"""
    #     try:
    #         gas_diff = max(0.0, (original_gas or 0.0) - (residual_gas or 0.0))
    #         val = cross_section * coal_density * tunneling_speed * gas_diff
    #         return max(0.0, float(val))
    #     except Exception as e:
    #         logger.error(f"计算落煤涌出量失败：{str(e)}", exc_info=True)
    #         return 0.0