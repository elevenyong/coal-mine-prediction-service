import os
import sys
from loguru import logger
import configparser
from datetime import datetime


def setup_logging(config_path="config.ini"):
    """
    初始化日志配置（全局唯一调用，建议在app.py启动时执行）

    目标：
    - 文件日志：详细（便于追溯/复盘）
    - 控制台日志：可选精简（只显示训练关键指标 + WARNING/ERROR）

    :param config_path: str，配置文件路径，默认"config.ini"
    :return: loguru.Logger，初始化后的全局日志对象
    """
    # Step 1: 读取日志相关配置
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    # 1) 日志目录：支持从配置读取，默认"logs"
    #    兼容你历史版本未配置 log_dir 的情况
    log_dir = config.get("Logging", "log_dir", fallback="logs")
    log_dir = log_dir.strip() if isinstance(log_dir, str) else "logs"
    if not log_dir:
        log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 2) 日志文件命名（按日期）
    log_filename = f"{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_filename)

    # 3) 保留天数（已有）
    keep_backups = config.getint("Logging", "keep_backups", fallback=5)

    # 4) 日志级别（新增读取，默认INFO）
    log_level = config.get("Logging", "log_level", fallback="INFO").upper().strip()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        log_level = "INFO"

    # 5) 控制台详细输出（已有）
    verbose_console = config.getboolean("Logging", "verbose_console", fallback=True)

    # 6) 训练日志简略显示（你配置里已有该开关）
    brief_training_logs = config.getboolean("Logging", "brief_training_logs", fallback=False)  # :contentReference[oaicite:3]{index=3}

    # Step 2: 清除loguru默认处理器（避免重复输出）
    logger.remove()

    # Step 3: 文件日志（详细：用于问题追溯）
    logger.add(
        sink=log_path,
        rotation="00:00",
        retention=f"{keep_backups} days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )

    # Step 4: 控制台日志（分层输出）
    # 4.1 控制台：永远输出 WARNING 及以上（不丢关键异常）
    warn_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sink=sys.stdout,
        format=warn_format,
        level="WARNING"
    )

    # 4.2 控制台：INFO 输出策略
    # - verbose_console=True：保留你原来的详细控制台INFO
    # - verbose_console=False：
    #     - brief_training_logs=True：只输出“简报INFO”（需要 logger.bind(brief=True)）
    #     - brief_training_logs=False：输出简化INFO（时间+级别+消息）
    if verbose_console:
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(
            sink=sys.stdout,
            format=console_format,
            level=log_level
        )
    else:
        if brief_training_logs:
            # 仅输出标记为 brief 的INFO（用于训练简报/关键指标）
            def _brief_filter(record):
                return bool(record["extra"].get("brief", False))

            brief_format = "{time:HH:mm:ss} | {message}"
            logger.add(
                sink=sys.stdout,
                level="INFO",
                format=brief_format,
                filter=_brief_filter
            )
        else:
            # 普通精简INFO（仍会输出所有INFO）
            console_format = (
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> - "
                "<level>{message}</level>"
            )
            logger.add(
                sink=sys.stdout,
                format=console_format,
                level=log_level
            )

    # Step 5: 记录日志初始化完成
    logger.info(f"日志系统初始化完成，日志文件路径：{log_path}，保留备份：{keep_backups}天，log_level={log_level}，verbose_console={verbose_console}，brief_training_logs={brief_training_logs}")
    return logger
