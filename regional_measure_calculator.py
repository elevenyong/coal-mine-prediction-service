"""
煤矿瓦斯风险预测系统 - 区域措施强度计算模块
包含：区域措施强度计算
"""
import json
from loguru import logger

from config_utils import ConfigUtils


class RegionalMeasureCalculator(ConfigUtils):
    """区域措施强度计算器"""

    def __init__(self, config_path="config.ini"):
        """
        初始化区域措施强度计算器
        """
        super().__init__(config_path)
        self._load_regional_measure_config()

    def _load_regional_measure_config(self):
        """
        私有方法：加载区域措施强度计算配置（[RegionalMeasureParams] section）
        参考标准：《煤矿井下瓦斯抽采工程设计规范》（吨煤钻孔量计算逻辑）
        """
        try:
            # 加载吨煤钻孔量计算所需配置
            self.default_coal_density = self._get_config_value(
                "RegionalMeasureParams", "default_coal_density", 1.4, is_float=True
            )
            self.min_coal_quantity = self._get_config_value(
                "RegionalMeasureParams", "min_coal_quantity", 0.001, is_float=True
            )
            logger.info(
                f"区域措施强度配置加载完成：默认煤密度={self.default_coal_density}t/m³，最小煤量阈值={self.min_coal_quantity}t"
            )
        except Exception as e:
            logger.warning(f"加载区域措施配置失败：{str(e)}，使用默认值")
            self._set_default_regional_params()

    def _set_default_regional_params(self):
        """私有方法：区域措施强度默认参数（配置缺失时兜底）"""
        self.default_coal_density = 1.4  # 默认煤密度（吨/立方米）
        self.min_coal_quantity = 0.001  # 最小煤量阈值（避免除以零）
        logger.debug(
            f"区域措施强度已设置默认参数：默认煤密度={self.default_coal_density}t/m³，最小煤量阈值={self.min_coal_quantity}t"
        )

    def calculate_regional_measure_strength(self, measures):
        """
        公开方法：计算区域措施强度（基于吨煤钻孔量）
        :param measures: list[dict]，区域措施列表，每个元素需包含：
            drill_total_length: float，单条措施钻孔总长度（米，必选）
            coal_seam_thickness: float，对应煤层厚度（米，必选）
            working_area: float，措施覆盖工作面积（平方米，必选）
            coal_density: float，煤层密度（吨/立方米，可选，默认用配置值）
        :return: list[dict]，补充strength字段的措施列表（strength=吨煤钻孔量，单位：m/t）
        """
        self._print_header("计算区域措施强度（吨煤钻孔量逻辑）")
        if not measures or not isinstance(measures, list):
            logger.warning("输入措施为空或格式错误，返回空列表")
            return []

        result = []
        for idx, measure in enumerate(measures):
            try:
                # 1. 提取并校验输入参数（处理缺失值+边界值）
                # 钻孔总长度（≥0，缺失设为0）
                drill_total_length = max(float(measure.get("drill_total_length", 0.0)), 0.0)
                # 煤层厚度（≥0.1，避免不合理值，缺失设为0.1）
                coal_seam_thickness = max(float(measure.get("coal_seam_thickness", 0.1)), 0.1)
                # 工作面积（≥1，避免不合理值，缺失设为1）
                working_area = max(float(measure.get("working_area", 1.0)), 1.0)
                # 煤密度（可选，默认用配置值，范围1.2-1.6）
                coal_density = measure.get("coal_density", self.default_coal_density)
                coal_density = min(max(float(coal_density), 1.2), 1.6)  # 约束合理范围

                # 2. 计算煤层煤量（吨）：厚度×面积×密度
                coal_quantity = coal_seam_thickness * working_area * coal_density
                # 兜底：煤量不小于最小阈值（避免除以零）
                coal_quantity = max(coal_quantity, self.min_coal_quantity)

                # 3. 计算吨煤钻孔量（区域措施强度）
                if drill_total_length == 0:
                    strength = 0.0  # 无钻孔时强度为0
                else:
                    strength = round(drill_total_length / coal_quantity, 4)  # 保留4位小数

                # 4. 补充结果字段
                measure["strength"] = strength
                result.append(measure)

                # 5. 日志输出（适配新逻辑）
                logger.debug(
                    f"措施[{idx}]强度计算完成：钻孔总长={drill_total_length}m，煤层厚度={coal_seam_thickness}m，"
                    f"工作面积={working_area}m²，煤密度={coal_density}t/m³，煤量={coal_quantity:.2f}t，"
                    f"吨煤钻孔量（强度）={strength}m/t"
                )
            except Exception as e:
                # 修复：计算失败时，将强度值设置为0并记录详细错误信息
                logger.error(f"计算措施[{idx}]强度失败：{str(e)}", exc_info=True)
                measure["strength"] = 0.0  # 设置为0而不是-1
                measure["calculation_error"] = str(e)  # 记录错误信息
                result.append(measure)
                logger.warning(f"措施[{idx}]强度计算失败，已设置为0.0，错误原因：{str(e)}")

        # 统计有效计算结果
        valid_count = sum(1 for m in result if m.get("strength", -1) != 0.0)
        error_count = sum(1 for m in result if "calculation_error" in m)
        valid_strengths = [m["strength"] for m in result if m.get("strength", -1) != 0.0]
        avg_strength = round(sum(valid_strengths) / max(valid_count, 1), 4) if valid_count > 0 else 0.0

        if error_count > 0:
            logger.warning(
                f"区域措施强度计算完成，但存在{error_count}条失败记录（已设置为0.0）：总措施数：{len(measures)}，有效计算数：{valid_count}，"
                f"平均吨煤钻孔量：{avg_strength}m/t，失败数：{error_count}"
            )
        else:
            logger.info(
                f"区域措施强度计算完成（吨煤钻孔量逻辑）：总措施数：{len(measures)}，有效计算数：{valid_count}，"
                f"平均吨煤钻孔量：{avg_strength}m/t"
            )

        self._print_result(
            f"计算完成（总措施数：{len(measures)}，有效：{valid_count}，平均强度：{avg_strength}m/t，失败：{error_count}）"
        )
        return result