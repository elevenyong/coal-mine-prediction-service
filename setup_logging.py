"""
煤矿瓦斯风险预测系统 - 日志配置模块
作用：初始化全局日志系统，支持：
  1. 按日轮转日志文件（凌晨00:00），保留指定天数备份并压缩
  2. 控制台输出分级（根据config.ini的verbose_console控制详细程度）
  3. 日志包含时间、级别、模块、函数、行号，便于问题追溯
依赖：loguru、configparser
"""
import os
import sys
from loguru import logger
import configparser
from datetime import datetime


def setup_logging(config_path="config.ini"):
    """
    初始化日志配置（全局唯一调用，建议在app.py启动时执行）

    :param config_path: str，配置文件路径，默认"config.ini"
    :return: loguru.Logger，初始化后的全局日志对象
    """
    # Step 1: 读取日志相关配置
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    # 日志目录配置（默认"logs/"）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    # 日志文件命名（按日期，如"20251015.log"）
    log_filename = f"{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_filename)
    # 日志保留天数（从config读取，默认5天）
    keep_backups = config.getint("Logging", "keep_backups", fallback=5)
    # 控制台详细输出开关（生产环境建议设为False）
    verbose_console = config.getboolean("Logging", "verbose_console", fallback=True)

    # Step 2: 清除loguru默认处理器（避免重复输出）
    logger.remove()

    # Step 3: 添加文件日志处理器（保留INFO及以上，用于问题追溯）
    logger.add(
        sink=log_path,
        rotation="00:00",  # 每天凌晨轮转日志文件
        retention=f"{keep_backups} days",  # 保留指定天数备份
        compression="zip",  # 旧日志压缩为zip，节省磁盘
        encoding="utf-8",
        enqueue=True,  # 异步写入，避免阻塞主流程
        backtrace=True,  # 包含异常堆栈回溯
        diagnose=True,  # 包含变量上下文（开发环境友好）
        level="INFO",  # 文件日志保留INFO及以上级别
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )

    # Step 4: 添加控制台日志处理器（按verbose_console控制输出格式）
    if verbose_console:
        # 详细格式：包含时间、级别、模块、函数、行号（开发环境）
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    else:
        # 简化格式：仅时间、级别、消息（生产环境）
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> - "
            "<level>{message}</level>"
        )
    logger.add(
        sink=sys.stdout,
        format=console_format,
        level="INFO"  # 控制台仅输出INFO及以上，避免DEBUG冗余
    )

    # Step 5: 记录日志初始化完成
    logger.info(f"日志系统初始化完成，日志文件路径：{log_path}，保留备份：{keep_backups}天")
    return logger