# file: source_prediction.py
"""
煤矿瓦斯风险预测系统 - 分源预测法模块
基于AQ1018-2006标准，独立计算瓦斯涌出量
"""

import numpy as np
from loguru import logger
from config_utils import ConfigUtils, timing_decorator


class SourcePredictionCalculator(ConfigUtils):
    """分源预测法计算器（独立模块）"""

    def __init__(self, config_path="config.ini"):
        super().__init__(config_path)
        self._load_source_prediction_params()

    def _load_source_prediction_params(self):
        """加载分源预测法配置"""
        try:
            # 从配置读取分源预测法参数
            self.source_params_config = {
                "default_coal_density": self._get_config_value(
                    "SourcePrediction", "default_coal_density", 1.4, is_float=True
                ),
                "base_initial_gas_emission": self._get_config_value(
                    "SourcePrediction", "base_initial_gas_emission", 0.01, is_float=True
                )
            }
            logger.info("分源预测法配置加载完成")
        except Exception as e:
            logger.warning(f"分源预测法配置加载失败，使用默认值：{str(e)}")
            self._set_default_source_params()

    def _set_default_source_params(self):
        """设置分源预测法默认参数"""
        self.source_params_config = {
            "default_coal_density": 1.4,  # 默认煤密度（吨/立方米）
            "base_initial_gas_emission": 0.01  # 默认初始瓦斯涌出强度
        }

    def calculate_gas_emission_source(self, data):
        """
        分源预测法计算瓦斯涌出量（独立接口）
        基于AQ1018-2006标准

        :param data: list[dict] / dict，输入数据，需包含：
            coal_thickness: float，煤层厚度（m）
            tunneling_speed: float，掘进速度（m/min）
            initial_gas_emission_strength: float，初始瓦斯涌出强度（可选）
            roadway_length: float，巷道长度（m）
            roadway_cross_section: float，巷道断面积（m²）
            coal_density: float，煤密度（t/m³，可选）
            original_gas_content: float，原始瓦斯含量（m³/t）
            residual_gas_content: float，残余瓦斯含量（m³/t）
        :return: list[dict]，补充瓦斯涌出量计算结果
        """
        self._print_header("分源预测法计算瓦斯涌出量")

        try:
            # 统一数据格式
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise TypeError("输入数据必须是字典或列表")

            results = []
            for idx, sample in enumerate(data):
                try:
                    # 提取并验证参数
                    params = self._extract_and_validate_params(sample)

                    # 计算煤壁瓦斯涌出量
                    q_wall = self._calculate_coal_wall_emission(
                        params["coal_thickness"],
                        params["tunneling_speed"],
                        params["initial_gas_emission_strength"],
                        params["roadway_length"]
                    )

                    # 计算落煤瓦斯涌出量
                    q_fallen = self._calculate_fallen_coal_emission(
                        params["roadway_cross_section"],
                        params["coal_density"],
                        params["tunneling_speed"],
                        params["original_gas_content"],
                        params["residual_gas_content"]
                    )

                    # 计算总瓦斯涌出量
                    total_q = q_wall + q_fallen

                    # 构建结果
                    result_sample = sample.copy()
                    result_sample.update({
                        "gas_emission_wall": round(q_wall, 4),
                        "gas_emission_fallen": round(q_fallen, 4),
                        "total_gas_emission": round(total_q, 4),
                        "calculation_method": "分源预测法(AQ1018-2006)"
                    })

                    results.append(result_sample)

                    logger.debug(
                        f"样本[{idx}]分源计算完成: "
                        f"煤壁瓦斯={q_wall:.4f}m³/min, "
                        f"落煤瓦斯={q_fallen:.4f}m³/min, "
                        f"总瓦斯={total_q:.4f}m³/min"
                    )

                except Exception as e:
                    logger.error(f"样本[{idx}]分源计算失败: {str(e)}")
                    # 返回包含错误信息的结果
                    error_result = sample.copy()
                    error_result.update({
                        "gas_emission_wall": 0.0,
                        "gas_emission_fallen": 0.0,
                        "total_gas_emission": 0.0,
                        "calculation_error": str(e),
                        "calculation_method": "分源预测法(计算失败)"
                    })
                    results.append(error_result)

            # 统计结果
            valid_count = sum(1 for r in results if "calculation_error" not in r)
            total_samples = len(results)

            self._print_result(
                f"分源预测计算完成: 总计{total_samples}条, "
                f"成功{valid_count}条, 失败{total_samples - valid_count}条"
            )

            return results

        except Exception as e:
            logger.error(f"分源预测法计算失败: {str(e)}", exc_info=True)
            raise

    def _extract_and_validate_params(self, sample):
        """提取并验证分源预测法参数"""
        required_params = [
            "coal_thickness",  # 煤层厚度
            "tunneling_speed",  # 掘进速度
            "roadway_length",  # 巷道长度
            "roadway_cross_section",  # 巷道断面积
            "original_gas_content",  # 原始瓦斯含量
            "residual_gas_content"  # 残余瓦斯含量
        ]

        # 检查必要参数
        missing_params = [p for p in required_params if p not in sample]
        if missing_params:
            raise ValueError(f"缺少必要参数: {missing_params}")

        # 提取参数，设置默认值
        params = {
            "coal_thickness": max(float(sample.get("coal_thickness", 0.0)), 0.0),
            "tunneling_speed": max(float(sample.get("tunneling_speed", 0.0)), 0.0),
            "roadway_length": max(float(sample.get("roadway_length", 0.0)), 0.0),
            "roadway_cross_section": max(float(sample.get("roadway_cross_section", 0.0)), 0.0),
            "original_gas_content": float(sample.get("original_gas_content", 0.0)),
            "residual_gas_content": float(sample.get("residual_gas_content", 0.0)),
            "coal_density": float(sample.get("coal_density", self.source_params_config["default_coal_density"])),
            "initial_gas_emission_strength": float(sample.get(
                "initial_gas_emission_strength",
                self.source_params_config["base_initial_gas_emission"]
            ))
        }

        # 参数合理性检查
        if params["coal_thickness"] == 0:
            logger.warning("煤层厚度为0，可能影响计算准确性")
        if params["tunneling_speed"] == 0:
            logger.warning("掘进速度为0，煤壁涌出量将设为0")

        return params

    def _calculate_coal_wall_emission(self, coal_thickness, tunneling_speed,
                                      initial_strength, roadway_length):
        """计算煤壁瓦斯涌出量（AQ1018-2006公式）"""
        try:
            if tunneling_speed <= 0:
                return 0.0

            # 防止除零
            tunneling_speed = max(tunneling_speed, 1e-9)

            # 计算公式: q_wall = M × v × q0 × (2√(L/v) - 1)
            # 其中: M-煤层厚度, v-掘进速度, q0-初始瓦斯涌出强度, L-巷道长度
            sqrt_term = np.sqrt(roadway_length / tunneling_speed)
            q_wall = coal_thickness * tunneling_speed * initial_strength * (2 * sqrt_term - 1)

            return max(0.0, float(q_wall))
        except Exception as e:
            logger.error(f"煤壁瓦斯计算失败: {str(e)}")
            return 0.0

    def _calculate_fallen_coal_emission(self, cross_section, coal_density,
                                        tunneling_speed, original_gas, residual_gas):
        """计算落煤瓦斯涌出量（AQ1018-2006公式）"""
        try:
            # 计算瓦斯含量差
            gas_diff = max(0.0, original_gas - residual_gas)

            # 计算公式: q_fallen = S × ρ × v × (W0 - Wc)
            # 其中: S-巷道断面积, ρ-煤密度, v-掘进速度, W0-原始瓦斯含量, Wc-残余瓦斯含量
            q_fallen = cross_section * coal_density * tunneling_speed * gas_diff

            return max(0.0, float(q_fallen))
        except Exception as e:
            logger.error(f"落煤瓦斯计算失败: {str(e)}")
            return 0.0

    def calculate_batch_gas_emission(self, data_list):
        """
        批量计算瓦斯涌出量（优化性能）

        :param data_list: list[dict]，批量数据
        :return: list[dict]，计算结果
        """
        return self.calculate_gas_emission_source(data_list)