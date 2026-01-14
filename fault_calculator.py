"""
煤矿瓦斯风险预测系统 - 断层影响计算模块
包含：断层影响系数批量计算、空间距离计算、断层数据查询
依赖：shapely（空间几何计算）、numpy、pandas
"""
import pandas as pd
import numpy as np
from loguru import logger
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

from config_utils import ConfigUtils


class FaultCalculator(ConfigUtils):
    """断层影响系数计算器"""

    def __init__(self, config_path="config.ini"):
        super().__init__(config_path)
        self._load_fault_config()

    def _load_fault_config(self):
        """
        私有方法：加载断层影响系数计算配置（[FaultParams] section）
        参考文献：《断层对瓦斯涌出影响的量化分析》
        """
        logger.debug("开始加载断层影响系数配置")

        try:
            self.base_strength = self._get_config_value("FaultParams", "base_strength", 0.5, is_float=True)
            self.distance_decay = self._get_config_value("FaultParams", "distance_decay", 0.8, is_float=True)
            self.fault_type_weight_1 = self._get_config_value("FaultParams", "fault_type_weight_1", 1.2, is_float=True)
            self.fault_type_weight_2 = self._get_config_value("FaultParams", "fault_type_weight_2", 0.8, is_float=True)
            self.fault_height_weight = self._get_config_value("FaultParams", "fault_height_weight", 0.15, is_float=True)
            self.fault_length_weight = self._get_config_value("FaultParams", "fault_length_weight", 0.1, is_float=True)
            self.inclination_weight = self._get_config_value("FaultParams", "inclination_weight", 0.01, is_float=True)
            self.max_influence_distance = self._get_config_value("FaultParams", "max_influence_distance", 50.0, is_float=True)

            logger.debug("断层影响系数配置加载完成")

        except Exception as e:
            logger.error(f"加载断层配置失败：{str(e)}", exc_info=True)
            # 重新抛出异常，让调用方知道配置加载失败
            raise

    def calculate_fault_influence_strength(self, data, db_utils):
        """
        公开方法：批量计算样本的断层影响系数（fault_influence_strength）

        :param data: list[dict] / pandas.DataFrame，输入数据，需包含：
            workface_id: int，工作面ID
            x_coord/y_coord/z_coord: float，掘进位置3D坐标
        :param db_utils: DBUtils实例，用于数据库查询
        :return: list[dict] / pandas.DataFrame，补充fault_influence_strength字段的数据
        :raises TypeError: 输入格式不支持时抛出
        :raises RuntimeError: 数据库查询失败时抛出
        """
        # 验证配置是否正确加载
        if not hasattr(self, 'base_strength'):
            logger.error("断层计算器配置未正确加载，base_strength属性不存在")
            raise RuntimeError("断层计算器配置未正确加载")

        logger.info("自动计算断层影响系数")

        # Step 1: 统一数据格式为DataFrame（方便批量处理）
        input_is_list = isinstance(data, list)
        if input_is_list:
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"输入格式不支持：{type(data)}，仅支持list[dict]或DataFrame")
        logger.debug(f"输入样本数：{len(df)}，已转换为DataFrame格式")

        # Step 2: 填充必要字段缺失值（避免计算失败）
        df["workface_id"] = df["workface_id"].fillna(-1).astype(int)
        coord_cols = ["x_coord", "y_coord", "z_coord"]
        for col in coord_cols:
            df[col] = df[col].fillna(0.0).astype(float)
            if df[col].isin([np.inf, -np.inf]).any():
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)
                logger.warning(f"检测到无效{col}值（无穷大），已替换为0.0")
        logger.debug("必要字段缺失值填充完成，确保计算安全")

        # Step 3: 按工作面分组计算（减少数据库查询次数，提升效率）
        fault_strength_list = []
        workface_groups = df.groupby("workface_id")

        # 记录数据库查询错误
        db_query_errors = []

        for workface_id, group in workface_groups:
            group_size = len(group)
            logger.debug(f"开始处理工作面ID {workface_id}：{group_size}条样本")

            if workface_id == -1:
                fault_strength_list.extend([self.base_strength] * group_size)
                logger.debug(f"工作面ID {workface_id}（未知）：系数设为基础值{self.base_strength}")
                continue

            try:
                # 修复：数据库查询失败时抛出异常，而不是静默使用基础值
                faults = self._get_faults_and_points(workface_id, db_utils)

                if not faults:
                    fault_strength_list.extend([self.base_strength] * group_size)
                    logger.debug(f"工作面ID {workface_id}：无有效断层，系数设为基础值{self.base_strength}")
                    continue

                for _, row in group.iterrows():
                    drill_x = row["x_coord"]
                    drill_y = row["y_coord"]
                    drill_z = row["z_coord"]
                    max_strength = self.base_strength

                    for fault in faults:
                        single_strength = self._calculate_single_fault_strength(
                            drill_x, drill_y, drill_z, fault
                        )
                        if single_strength > max_strength:
                            max_strength = single_strength

                    fault_strength_list.append(round(max_strength, 2))

            except Exception as e:
                # 记录数据库查询错误，但继续处理其他工作面
                error_msg = f"工作面ID {workface_id} 的断层数据查询失败：{str(e)}"
                logger.error(error_msg, exc_info=True)
                db_query_errors.append(error_msg)
                # 设置为错误标志值，而不是基础值
                fault_strength_list.extend([-999.0] * group_size)

        # Step 4: 将计算结果添加到数据中并还原输入格式
        if len(fault_strength_list) == len(df):
            df["fault_influence_strength"] = fault_strength_list

            # 检查是否有数据库查询错误
            if db_query_errors:
                error_indices = df[df["fault_influence_strength"] == -999.0].index.tolist()
                error_count = len(error_indices)
                logger.error(f"断层影响系数计算存在{error_count}条数据库查询错误，已标记为-999.0")
                logger.error(f"错误详情：{db_query_errors[:3]}")  # 只显示前3个错误

                # 将错误标记值替换为基础值，避免影响后续计算
                df.loc[error_indices, "fault_influence_strength"] = self.base_strength
                logger.warning(f"已将{error_count}条错误记录的断层系数替换为基础值{self.base_strength}")

            # 计算有效结果的统计信息
            valid_results = df[df["fault_influence_strength"] != -999.0]["fault_influence_strength"]
            if len(valid_results) > 0:
                avg_strength = round(valid_results.mean(), 2)
                max_strength = valid_results.max()
                logger.info(
                    f"断层影响系数计算完成：样本数：{len(df)}，有效计算：{len(valid_results)}，平均值：{avg_strength}，最大值：{max_strength}"
                )
            else:
                logger.warning("断层影响系数计算完成，但所有结果均为错误标记值")
                avg_strength = self.base_strength
                max_strength = self.base_strength
        else:
            logger.error(
                f"断层系数计算结果长度不匹配：预期{len(df)}条，实际{len(fault_strength_list)}条"
            )
            raise ValueError("断层影响系数计算结果长度异常")

        if input_is_list:
            return df.to_dict("records")
        else:
            return df

    def _get_faults_and_points(self, workface_id, db_utils):
        """
        私有方法：按工作面ID查询断层及对应的组成点（过滤无效断层）

        :param workface_id: int，工作面ID
        :param db_utils: DBUtils实例
        :return: list[dict]，有效断层列表（每个元素含points字段：组成点列表）
        :raises Exception: 数据库查询失败时抛出异常
        """
        try:
            # 修复：数据库查询失败时抛出异常，而不是返回空列表
            faults = db_utils.get_faults_by_workface(workface_id)
            valid_faults = []
            for fault in faults:
                fault_id = fault["id"]
                try:
                    points = db_utils.get_fault_points(fault_id)
                    if len(points) >= 2:
                        fault["points"] = points
                        valid_faults.append(fault)
                        logger.debug(
                            f"断层{fault_id}（{fault['name']}）验证通过：组成点{len(points)}个，断距{fault.get('fault_height', 0)}m"
                        )
                    else:
                        logger.warning(
                            f"断层{fault_id}（{fault['name']}）无效：组成点{len(points)}个（需≥2个），已过滤"
                        )
                except Exception as e:
                    logger.error(f"查询断层{fault_id}的组成点失败：{str(e)}")
                    # 继续处理其他断层，不中断整个流程
            logger.debug(f"工作面{workface_id}共查询到有效断层：{len(valid_faults)}个")
            return valid_faults
        except Exception as e:
            logger.error(f"查询工作面{workface_id}的断层信息失败：{str(e)}", exc_info=True)
            raise RuntimeError(f"数据库查询失败：无法获取工作面{workface_id}的断层数据") from e

    def _calculate_shortest_distance_to_fault(self, point_x, point_y, point_z, fault_points):
        """
        私有方法：计算掘进位置（3D点）到断层（3D线段）的最短距离
        :param point_x: float，掘进位置X坐标（m）
        :param point_y: float，掘进位置Y坐标（m）
        :param point_z: float，掘进位置Z坐标（高程，m）
        :param fault_points: list[dict]，断层组成点（含floor_coordinate_x/y/z）
        :return: float，最短距离（m），异常时返回max_influence_distance
        """
        try:
            fault_shapely_points = [
                Point(
                    p["floor_coordinate_x"],
                    p["floor_coordinate_y"],
                    p["floor_coordinate_z"]
                )
                for p in fault_points
            ]
            fault_line = LineString(fault_shapely_points)
            drill_point = Point(point_x, point_y, point_z)
            nearest_point = nearest_points(fault_line, drill_point)[0]
            distance = drill_point.distance(nearest_point)
            logger.debug(
                f"掘进位置({point_x:.2f},{point_y:.2f},{point_z:.2f})到断层的最短距离：{distance:.2f}m"
            )
            return float(distance)
        except Exception as e:
            logger.error(f"计算断层距离失败：{str(e)}", exc_info=True)
            return self.max_influence_distance

    def _calculate_single_fault_strength(self, drill_x, drill_y, drill_z, fault):
        """
        私有方法：计算单个断层对掘进位置的影响系数（核心公式实现）
        :param drill_x: float，掘进位置X坐标
        :param drill_y: float，掘进位置Y坐标
        :param drill_z: float，掘进位置Z坐标
        :param fault: dict，断层信息（含id、points、fault_type等属性）
        :return: float，单个断层影响系数（范围0.0~2.0）
        """
        try:
            fault_id = fault.get("id", "未知")
            fault_type = fault.get("fault_type", 1)
            fault_height = max(fault.get("fault_height", 0.0), 0.0)
            fault_length = max(fault.get("length", 0.0), 0.0)
            fault_inclination = min(max(fault.get("inclination", 0.0), 0.0), 180.0)
            fault_influence_scope = fault.get("influence_scope", self.max_influence_distance)
            if fault_influence_scope <= 0:
                fault_influence_scope = self.max_influence_distance
                logger.warning(f"断层{fault_id}影响范围无效（≤0），使用默认{self.max_influence_distance}m")
            fault_points = fault.get("points", [])
            if len(fault_points) < 2:
                logger.warning(f"断层{fault_id}组成点不足2个，影响系数设为0.0")
                return 0.0
            distance_to_fault = self._calculate_shortest_distance_to_fault(
                drill_x, drill_y, drill_z, fault_points
            )

            if distance_to_fault >= fault_influence_scope:
                logger.debug(
                    f"断层{fault_id}：距离{distance_to_fault:.2f}m≥影响范围{fault_influence_scope}m，距离因子=0.0"
                )
                return 0.0

            distance_factor = (fault_influence_scope - distance_to_fault) / fault_influence_scope
            distance_factor *= (self.distance_decay ** distance_to_fault)
            distance_factor = round(distance_factor, 4)

            type_weight = self.fault_type_weight_1 if fault_type == 1 else self.fault_type_weight_2

            height_factor = 1.0 + self.fault_height_weight * fault_height
            height_factor = min(height_factor, 1.5)

            length_factor = 1.0 + self.fault_length_weight * fault_length
            length_factor = min(length_factor, 1.5)

            inclination_factor = 1.0 + self.inclination_weight * abs(fault_inclination - 90)
            inclination_factor = round(inclination_factor, 4)

            property_factor = type_weight * height_factor * length_factor * inclination_factor
            property_factor = round(property_factor, 4)

            final_strength = self.base_strength * distance_factor * property_factor
            final_strength = min(final_strength, 2.0)
            final_strength = round(final_strength, 2)

            logger.debug(
                f"断层{fault_id}影响系数计算完成：距离因子：{distance_factor}，属性因子：{property_factor}，最终系数：{final_strength}"
            )
            return final_strength
        except Exception as e:
            logger.error(f"计算断层{fault.get('id', '未知')}影响系数失败：{str(e)}", exc_info=True)
            return 0.0