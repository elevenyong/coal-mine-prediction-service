"""
煤矿瓦斯风险预测系统 - 配置工具模块
包含：全局辅助函数、配置读取工具、日志打印方法
"""
import time
import os
from datetime import datetime
from functools import wraps
from loguru import logger
import configparser


# -------------------- 全局辅助函数 --------------------
def now_str():
    """
    辅助函数：生成标准时间字符串（用于日志标记、数据时间戳）
    :return: str，格式如"2025-10-15 16:30:00"
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def timing_decorator(func):
    """
    装饰器：统计函数执行耗时，输出INFO级日志（用于train/predict等核心方法）
    :param func: function，被装饰的函数
    :return: function，包装后的函数（保留原函数元信息）
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"【耗时统计】{func.__name__} 执行完成，耗时 {duration:.2f} 秒")
        return result
    return wrapper

# +++ 统一异常处理装饰器
def error_handler_decorator(func):
    """
    装饰器：统一异常处理逻辑，标准化日志输出格式
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 记录详细异常信息（含堆栈）
            logger.error(f"【方法异常】{func.__name__} 执行失败: {str(e)}", exc_info=True)
            # 返回标准化错误结果（便于上层统一处理）
            return {"status": "error", "message": str(e), "method": func.__name__}
    return wrapper


class ConfigUtils:
    """配置工具类，提供通用的配置读取和日志打印方法"""
    def __init__(self, config_path="config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self._load_config_with_merge()

    def _load_config_with_merge(self):
        """支持配置合并的加载方法"""
        # 检查文件是否存在
        if not os.path.exists(self.config_path):
            logger.error(f"配置文件不存在: {os.path.abspath(self.config_path)}")
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        # 检查文件权限
        if not os.access(self.config_path, os.R_OK):
            logger.error(f"配置文件不可读: {os.path.abspath(self.config_path)}")
            raise PermissionError(f"配置文件不可读: {self.config_path}")
        # 尝试解析配置
        try:
            # 首先读取基础配置
            base_config_path = "config.ini"
            if os.path.exists(base_config_path):
                self.config.read(base_config_path, encoding="utf-8")
                logger.debug(f"成功读取基础配置文件: {base_config_path}")
            else:
                logger.warning(f"基础配置文件不存在: {base_config_path}")
            # 然后读取当前配置（覆盖基础配置）
            if self.config_path != "config.ini" and os.path.exists(self.config_path):
                read_files = self.config.read(self.config_path, encoding="utf-8")
                if read_files:
                    logger.debug(f"成功合并配置文件: {self.config_path}")
                else:
                    logger.warning(f"无法读取配置文件: {self.config_path}")
            elif self.config_path == "config.ini":
                logger.debug("使用基础配置文件")
            # 检查关键配置项
            required_sections = ["Model", "ModelEval", "Features"]  # 核心配置段
            for section in required_sections:
                if section not in self.config.sections():
                    logger.warning(f"配置文件缺少核心段: [{section}]")
        except configparser.Error as e:
            logger.error(f"配置文件解析错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"读取配置文件失败: {str(e)}")
            raise

    def _print_header(self, title):
        """私有方法：打印控制台标题（仅当verbose_console=True时显示）"""
        if self.config.getboolean("Logging", "verbose_console", fallback=True):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {title}")
            print("-" * 40)

    def _print_step(self, msg):
        """私有方法：打印控制台步骤信息（缩进对齐）"""
        if self.config.getboolean("Logging", "verbose_console", fallback=True):
            print(f"│ {msg}")

    def _print_result(self, msg):
        """私有方法：打印控制台结果信息（带结果标识）"""
        if self.config.getboolean("Logging", "verbose_console", fallback=True):
            print(f"├─ 结果：{msg}")

    def _get_config_value(self, section, key, default, is_int=False, is_float=False,
                          min_value=None, max_value=None):
        """
        私有方法：安全读取配置值（支持类型转换和范围校验，解决问题1、6）

        :param section: str，配置文件section名
        :param key: str，配置项key
        :param default: 默认值（任意类型）
        :param is_int: bool，是否转换为int类型
        :param is_float: bool，是否转换为float类型
        :param min_value: 最小值限制（数值类型时有效）
        :param max_value: 最大值限制（数值类型时有效）
        :return: 配置值（按类型转换后）
        """
        try:
            if not self.config.has_section(section):
                logger.warning(f"配置section不存在: [{section}]")
                return default
            if not self.config.has_option(section, key):
                logger.warning(f"配置选项不存在: [{section}]->{key}，使用默认值{default}")
                return default
            if is_int:
                value = self.config.getint(section, key)
            elif is_float:
                value = self.config.getfloat(section, key)
            else:
                value = self.config.get(section, key)
            # 数值范围校验（解决问题6）
            if (min_value is not None or max_value is not None) and isinstance(value, (int, float)):
                if min_value is not None and value < min_value:
                    logger.warning(f"配置[{section}]->{key}={value}小于最小值{min_value}，使用默认值{default}")
                    return default
                if max_value is not None and value > max_value:
                    logger.warning(f"配置[{section}]->{key}={value}大于最大值{max_value}，使用默认值{default}")
                    return default
            logger.debug(f"读取配置: [{section}]->{key} = {value}")
            return value
        except ValueError as e:
            logger.warning(f"配置[{section}]->{key}类型转换错误：{str(e)}，使用默认值{default}")
            return default
        except Exception as e:
            logger.warning(f"配置[{section}]->{key}读取错误：{str(e)}，使用默认值{default}")
            return default