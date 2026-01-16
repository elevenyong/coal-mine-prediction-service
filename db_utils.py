"""
煤矿瓦斯风险预测系统 - 数据库工具模块（最新版：分表写入 + 保留原表不删列）

设计目标（分表而不删列）：
1) t_prediction_parameters：仅保存“原始样本 + 目标值 + 最少索引”
2) t_source_prediction_inputs：分源预测法输入参数专表（按 pred_id 关联）
3) t_feature_cache：时空增强统计特征专表（按 pred_id 关联）
4) 不修改/不删除 t_prediction_parameters 的既有列；只是写入策略改变
5) 兼容外部事务（train流程中由外部控制 commit/rollback）
"""

import configparser
import os
import time
from datetime import timedelta
from typing import Dict, Optional, Set, List, Any

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection


class DBUtils:
    """
    数据库工具类：表初始化、动态字段、分表写入、断层数据查询、训练记录写入
    """

    # 表名常量
    TABLE_PRED = "t_prediction_parameters"
    TABLE_TRAIN = "t_training_records"
    TABLE_FAULT = "t_geo_fault"
    TABLE_FAULT_POINT = "t_coal_point"
    TABLE_SOURCE = "t_source_prediction_inputs"
    TABLE_FEATURE = "t_feature_cache"

    def __init__(self, config_path: str = "config.ini"):
        logger.warning(f"[诊断] 正在加载 DBUtils 文件：{__file__}")
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding="utf-8")

        # Step 1: 读取数据库配置
        self.db_config = self._read_db_config()

        # Step 2: 创建 engine
        self.engine = self._build_engine(self.db_config)

        # Step 3: 加载动态字段配置（只针对 t_prediction_parameters 动态加列）
        self.dynamic_columns: Dict[str, str] = {}
        self._load_dynamic_columns()

        # Step 4: 初始化表结构
        self._init_tables()

        # Step 5: t_prediction_parameters 字段缓存（10分钟）
        self._pred_table_cols_cache: Optional[Set[str]] = None
        self._pred_table_cols_cache_ts: float = 0.0
        self._pred_table_cols_cache_ttl: int = 600  # 秒

        # Step 5.1: t_feature_cache 字段缓存（10分钟）
        self._feature_table_cols_cache: Optional[Set[str]] = None
        self._feature_table_cols_cache_ts: float = 0.0
        self._feature_table_cols_cache_ttl: int = 600  # 秒

        logger.debug("DBUtils 初始化完成")

    # -------------------------------------------------------------------------
    # 连接与配置
    # -------------------------------------------------------------------------
    def _read_db_config(self) -> Dict[str, str]:
        try:
            host = self.config.get("Database", "host")
            port = self.config.get("Database", "port")
            user = self.config.get("Database", "user")
            password = os.getenv("DB_PASSWORD") or self.config.get("Database", "password")
            db_name = self.config.get("Database", "db_name")
            charset = self.config.get("Database", "charset", fallback="utf8mb4")

            return {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "db": db_name,
                "charset": charset
            }
        except Exception as e:
            logger.error(f"读取数据库配置失败：{str(e)}")
            raise

    def _build_engine(self, db_conf: Dict[str, str]):
        db_url = (
            f"mysql+pymysql://{db_conf['user']}:{db_conf['password']}@"
            f"{db_conf['host']}:{db_conf['port']}/{db_conf['db']}?charset={db_conf['charset']}"
        )
        return create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            future=True
        )

    def _get_connection(self) -> Connection:
        """
        获取 SQLAlchemy Connection（由调用方负责关闭）
        """
        return self.engine.connect()

    def reload_config(self, config_path: Optional[str] = None) -> bool:
        """
        重载配置（会重建 engine）。一般不建议在训练过程中调用。
        """
        try:
            if config_path:
                self.config_path = config_path
                self.config.read(config_path, encoding="utf-8")

            new_db_conf = self._read_db_config()
            # 如果没有变化则跳过
            if all(self.db_config.get(k) == new_db_conf.get(k) for k in self.db_config.keys()):
                logger.info("数据库配置未发生变化，跳过重载")
                return True

            new_engine = self._build_engine(new_db_conf)
            # 测试连接
            with new_engine.connect() as c:
                c.execute(text("SELECT 1"))

            # 切换 engine
            old_engine = self.engine
            self.engine = new_engine
            self.db_config = new_db_conf

            try:
                old_engine.dispose()
            except Exception:
                pass

            # 重新加载动态字段并初始化表
            self._load_dynamic_columns()
            self._init_tables()

            # 清空缓存
            self._pred_table_cols_cache = None
            self._pred_table_cols_cache_ts = 0.0

            logger.info("数据库配置重载完成")
            return True

        except Exception as e:
            logger.error(f"数据库配置重载失败：{str(e)}", exc_info=True)
            return False

    # -------------------------------------------------------------------------
    # 动态字段（仅作用于 t_prediction_parameters）
    # -------------------------------------------------------------------------
    def _load_dynamic_columns(self):
        """
        从配置读取动态字段：Features.dynamic_db_columns
        格式：measurement_date:DATE;work_stage:VARCHAR(20);xxx:FLOAT
        """
        self.dynamic_columns = {}
        try:
            dynamic_str = self.config.get("Features", "dynamic_db_columns", fallback="").strip()
            if not dynamic_str:
                return

            for part in dynamic_str.split(";"):
                part = part.strip()
                if not part or ":" not in part:
                    continue
                k, v = part.split(":", 1)
                self.dynamic_columns[k.strip()] = v.strip()

            logger.debug(f"动态字段配置加载完成：{self.dynamic_columns}")
        except Exception as e:
            logger.warning(f"动态字段配置加载失败：{str(e)}（将忽略动态字段）")
            self.dynamic_columns = {}

    def _add_dynamic_columns(self, conn: Connection):
        """
        为 t_prediction_parameters 增加动态字段（不存在才加）
        """
        if not self.dynamic_columns:
            return

        try:
            rows = conn.execute(text("""
                SELECT COLUMN_NAME
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :table_name
            """), {"table_name": self.TABLE_PRED}).fetchall()
            existing = {r[0] for r in rows} if rows else set()

            for col_name, col_type in self.dynamic_columns.items():
                if col_name in existing:
                    continue
                alter_sql = text(f"""
                    ALTER TABLE {self.TABLE_PRED}
                    ADD COLUMN {col_name} {col_type} COMMENT '动态字段'
                """)
                conn.execute(alter_sql)
                logger.debug(f"动态添加字段：{col_name} {col_type}")

            # 动态字段变化后刷新缓存
            self._pred_table_cols_cache = None
            self._pred_table_cols_cache_ts = 0.0

        except Exception as e:
            logger.error(f"动态添加字段失败：{str(e)}", exc_info=True)

    def _add_feature_cache_columns(self, conn: Connection):
        """
        为 t_feature_cache（TABLE_FEATURE）补齐增强特征列（不存在才加）
        目的：保证训练/评估/预测阶段对 rolling_mean / hist_fused 等增强列一致可用。
        """
        try:
            rows = conn.execute(text("""
                SELECT COLUMN_NAME
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :table_name
            """), {"table_name": self.TABLE_FEATURE}).fetchall()
            existing = {r[0] for r in rows} if rows else set()

            # 需要补齐的列（只补你当前系统会用到的增强列）
            to_add = {
                "gas_emission_velocity_q_rolling_mean": "FLOAT",
                "gas_emission_velocity_q_hist_fused": "FLOAT",
            }

            for col_name, col_type in to_add.items():
                if col_name in existing:
                    continue
                conn.execute(text(f"""
                    ALTER TABLE {self.TABLE_FEATURE}
                    ADD COLUMN {col_name} {col_type} COMMENT '增强特征自动迁移'
                """))
                logger.debug(f"特征缓存表自动迁移：新增字段 {col_name} {col_type}")
        except Exception as e:
            logger.warning(f"特征缓存表字段自动迁移失败（已忽略）：{repr(e)}", exc_info=True)

    # -------------------------------------------------------------------------
    # 表结构初始化（含分表）
    # -------------------------------------------------------------------------
    def _init_tables(self):
        """
        初始化表结构（不存在则创建）
        """
        conn = self._get_connection()
        trans = conn.begin()
        try:
            # Step 1: 核心样本表（保留原表，不删列）
            conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_PRED} (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
                working_face VARCHAR(50) NOT NULL COMMENT '工作面名称',
                workface_id INT COMMENT '工作面ID',
                roadway VARCHAR(50) COMMENT '巷道名称',
                roadway_id VARCHAR(20) COMMENT '巷道ID',

                -- 空间坐标
                x_coord FLOAT COMMENT 'X坐标（m）',
                y_coord FLOAT COMMENT 'Y坐标（m）',
                z_coord FLOAT COMMENT 'Z坐标（高程，m）',
                distance_from_entrance FLOAT COMMENT '距入口距离（m）',

                -- 钻孔信息
                borehole_id VARCHAR(50) COMMENT '钻孔编号',
                drilling_depth FLOAT COMMENT '钻孔深度（m）',

                -- 时间信息
                measurement_date DATE COMMENT '测量日期',

                -- 工作面推进信息
                distance_to_face FLOAT COMMENT '距采面距离（m）',
                face_advance_distance FLOAT COMMENT '工作面推进距离（m）',
                advance_rate FLOAT COMMENT '推进速率（m/天）',

                -- 基础特征
                coal_thickness FLOAT COMMENT '煤层厚度（m）',
                fault_influence_strength FLOAT COMMENT '断层影响系数',
                regional_measure_strength INT COMMENT '区域措施强度',

                -- 分源预测法参数（原表保留，但分表方案默认不再写入这些列）
                tunneling_speed FLOAT COMMENT '掘进速度（m/min）',
                roadway_length FLOAT COMMENT '巷道长度（m）',
                initial_gas_emission_strength FLOAT COMMENT '初始瓦斯涌出强度',
                roadway_cross_section FLOAT COMMENT '巷道断面积（m²）',
                coal_density FLOAT COMMENT '煤密度（t/m³）',
                original_gas_content FLOAT COMMENT '原始瓦斯含量（m³/t）',
                residual_gas_content FLOAT COMMENT '残余瓦斯含量（m³/t）',

                -- 预测目标（q、S值）
                drilling_cuttings_s FLOAT COMMENT '钻屑量S（kg/m）',
                gas_emission_velocity_q FLOAT COMMENT '瓦斯涌出速度q（L/min·m）',

                -- 历史统计与时空字段（原表保留，分表方案默认不再写入）
                gas_emission_q_historical_mean FLOAT COMMENT '历史瓦斯涌出量均值',
                drilling_cuttings_s_historical_mean FLOAT COMMENT '历史钻屑量均值',
                gas_emission_velocity_q_historical_mean FLOAT COMMENT '历史瓦斯涌出速度均值',
                coord_hash VARCHAR(100) COMMENT '坐标哈希值',
                spatiotemporal_group VARCHAR(100) COMMENT '时空分组标识',

                create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',

                INDEX idx_working_face (working_face),
                INDEX idx_measurement_date (measurement_date),
                INDEX idx_distance_to_face (distance_to_face)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='瓦斯预测参数表（核心样本表）';
            """))
            logger.debug(f"表 {self.TABLE_PRED} 初始化完成（已存在则跳过）")

            # Step 2: 训练记录表
            conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_TRAIN} (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
                sample_count INT COMMENT '本次训练样本数',
                total_samples INT COMMENT '累计样本数',
                train_mode VARCHAR(50) COMMENT '训练模式（全量/增量）',
                status VARCHAR(20) COMMENT '训练状态（success/warning/error）',
                message VARCHAR(255) COMMENT '训练信息（成功/失败原因）',
                duration FLOAT COMMENT '训练耗时（秒）',
                train_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '训练时间',
                INDEX idx_train_time (train_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型训练记录表';
            """))
            logger.debug(f"表 {self.TABLE_TRAIN} 初始化完成（已存在则跳过）")

            # Step 3: 断层表
            conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_FAULT} (
                id INT NOT NULL AUTO_INCREMENT COMMENT '自增主键',
                workface_id INT NOT NULL COMMENT '工作面ID',
                name VARCHAR(50) NOT NULL COMMENT '断层名称',
                length DECIMAL(10,2) DEFAULT NULL COMMENT '断层长度（m）',
                azimuth DECIMAL(10,2) DEFAULT NULL COMMENT '方位角（°）',
                inclination DECIMAL(10,2) DEFAULT NULL COMMENT '倾角（°）',
                fault_height DECIMAL(10,2) DEFAULT NULL COMMENT '断距（m）',
                fault_type TINYINT(4) DEFAULT NULL COMMENT '断层类型（1=正断层，2=逆断层）',
                influence_scope DECIMAL(10,2) NOT NULL DEFAULT 0.00 COMMENT '影响范围（m）',
                according_to VARCHAR(255) DEFAULT NULL COMMENT '断层依据',
                create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                PRIMARY KEY (id),
                KEY idx_workface_id (workface_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='断层基础信息表';
            """))
            logger.debug(f"表 {self.TABLE_FAULT} 初始化完成（已存在则跳过）")

            # Step 4: 断层点表
            conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_FAULT_POINT} (
                id INT NOT NULL AUTO_INCREMENT COMMENT '自增主键',
                geofault_id INT NOT NULL COMMENT '关联断层ID（t_geo_fault.id）',
                floor_coordinate_x DECIMAL(10,2) NOT NULL COMMENT 'X坐标（m）',
                floor_coordinate_y DECIMAL(10,2) NOT NULL COMMENT 'Y坐标（m）',
                floor_coordinate_z DECIMAL(10,2) DEFAULT 0.00 COMMENT 'Z坐标（m）',
                point_order INT NOT NULL COMMENT '点顺序',
                workface_id INT NOT NULL COMMENT '工作面ID',
                create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                PRIMARY KEY (id),
                KEY fk_geofault (geofault_id),
                CONSTRAINT fk_geofault FOREIGN KEY (geofault_id)
                    REFERENCES {self.TABLE_FAULT} (id) ON DELETE CASCADE,
                CONSTRAINT uk_point_order UNIQUE KEY (geofault_id, point_order)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='断层组成点集合表';
            """))
            logger.debug(f"表 {self.TABLE_FAULT_POINT} 初始化完成（已存在则跳过）")

            # Step 5: 分源预测输入专表（新增）
            conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_SOURCE} (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
                pred_id INT NOT NULL COMMENT '关联t_prediction_parameters.id',

                -- 冗余索引字段
                working_face VARCHAR(50) NOT NULL COMMENT '工作面名称',
                workface_id INT COMMENT '工作面ID',
                work_stage VARCHAR(20) COMMENT '工况阶段（掘进/回采）',
                roadway VARCHAR(50) COMMENT '巷道名称（掘进真实/回采占位）',
                roadway_id VARCHAR(20) COMMENT '巷道ID（掘进真实/回采占位）',
                measurement_date DATE COMMENT '测量日期',
                x_coord FLOAT COMMENT 'X坐标（m）',
                y_coord FLOAT COMMENT 'Y坐标（m）',
                z_coord FLOAT COMMENT 'Z坐标（m）',
                drilling_depth FLOAT COMMENT '钻孔深度（m）',
                distance_to_face FLOAT COMMENT '距采面距离（m）',

                -- 分源预测法参数
                tunneling_speed FLOAT COMMENT '掘进速度（m/min）',
                roadway_length FLOAT COMMENT '巷道长度（m）',
                initial_gas_emission_strength FLOAT COMMENT '初始瓦斯涌出强度',
                roadway_cross_section FLOAT COMMENT '巷道断面积（m²）',
                coal_density FLOAT COMMENT '煤密度（t/m³）',
                original_gas_content FLOAT COMMENT '原始瓦斯含量（m³/t）',
                residual_gas_content FLOAT COMMENT '残余瓦斯含量（m³/t）',

                create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                INDEX idx_pred_id (pred_id),
                INDEX idx_workface_date (workface_id, measurement_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分源预测输入参数专表';
            """))
            logger.debug(f"表 {self.TABLE_SOURCE} 初始化完成（已存在则跳过）")

            # Step 6: 时空增强缓存专表（新增）
            conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_FEATURE} (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
                pred_id INT NOT NULL COMMENT '关联t_prediction_parameters.id',

                -- 冗余索引字段
                working_face VARCHAR(50) NOT NULL COMMENT '工作面名称',
                workface_id INT COMMENT '工作面ID',
                work_stage VARCHAR(20) COMMENT '工况阶段（掘进/回采）',
                measurement_date DATE COMMENT '测量日期',
                x_coord FLOAT COMMENT 'X坐标（m）',
                y_coord FLOAT COMMENT 'Y坐标（m）',
                z_coord FLOAT COMMENT 'Z坐标（m）',
                distance_to_face FLOAT COMMENT '距采面距离（m）',

                -- 时空标识/分桶
                coord_hash VARCHAR(100) COMMENT '坐标哈希值',
                spatiotemporal_group VARCHAR(100) COMMENT '时空分组标识',
                distance_to_face_bucket FLOAT COMMENT '距采面距离分桶（m）',

                -- 时间增强
                days_since_start INT COMMENT '距开始天数',
                days_in_workface INT COMMENT '工作面内天数',
                distance_time_interaction FLOAT COMMENT '距离-时间交互项',

                -- 历史统计/趋势
                gas_emission_q_historical_mean FLOAT COMMENT '历史瓦斯涌出量均值',
                drilling_cuttings_s_historical_mean FLOAT COMMENT '历史钻屑量均值',
                gas_emission_velocity_q_historical_mean FLOAT COMMENT '历史瓦斯涌出速度均值',
                gas_emission_velocity_q_rolling_mean FLOAT COMMENT '瓦斯涌出速度滑动均值（历史窗口）',
                gas_emission_velocity_q_hist_fused FLOAT COMMENT '瓦斯涌出速度历史融合（均值+滑动均值）',
                gas_emission_q_trend FLOAT COMMENT '瓦斯涌出量趋势',
                drilling_cuttings_s_trend FLOAT COMMENT '钻屑量趋势',
                gas_emission_velocity_q_trend FLOAT COMMENT '瓦斯涌出速度趋势',

                -- 其他增强字段
                advance_rate FLOAT COMMENT '推进速率（m/天）',

                create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                INDEX idx_pred_id (pred_id),
                INDEX idx_workface_date (workface_id, measurement_date),
                INDEX idx_coord_hash (coord_hash),
                INDEX idx_spatiotemporal_group (spatiotemporal_group)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='时空增强特征缓存专表';
            """))
            logger.debug(f"表 {self.TABLE_FEATURE} 初始化完成（已存在则跳过）")

            # Step 6.1: 自动迁移（表已存在时补齐增强列）
            self._add_feature_cache_columns(conn)

            # Step 7: 动态字段添加（只针对核心表）
            self._add_dynamic_columns(conn)

            trans.commit()
            logger.debug("所有表结构初始化完成（已提交）")

        except Exception as e:
            trans.rollback()
            logger.error(f"表结构初始化失败：{str(e)}，已回滚", exc_info=True)
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 统一列清单生成器（JOIN t_prediction_parameters + t_feature_cache）
    # -------------------------------------------------------------------------
    def _safe_col_name(self, col: Any) -> Optional[str]:
        '''
        列名清洗（用于拼 SQL 列清单）：
        - 仅允许字母/数字/下划线
        - 去掉空白
        - 返回 None 表示非法
        '''
        try:
            if col is None:
                return None
            c = str(col).strip()
            if not c:
                return None
            # 永久禁止内部临时列/merge后缀列
            if c.startswith("_") or c.endswith(("_x", "_y")):
                return None
            # 仅允许 [A-Za-z0-9_]
            if not c.replace("_", "").isalnum():
                return None
            return c
        except Exception:
            return None

    def _normalize_col_list(self, cols) -> List[str]:
        out = []
        if not cols:
            return out
        try:
            for c in cols:
                cc = self._safe_col_name(c)
                if cc and cc not in out:
                    out.append(cc)
        except Exception:
            return out
        return out

    # -------------------------------------------------------------------------
    # schema 字段读取（重要：请勿重复定义同名函数，否则后者会覆盖前者）
    # -------------------------------------------------------------------------
    def _get_prediction_table_columns(self, conn: Optional[Connection] = None) -> Set[str]:
        """
        获取 t_prediction_parameters 字段集合（带缓存，10分钟过期）
        - 支持传入 conn（同事务读取未提交数据；也避免重复开连接）
        - 强约束：失败返回空集合，调用方自行兜底
        """
        try:
            now_ts = time.time()
            if (getattr(self, "_pred_table_cols_cache", None) is not None and
                    (now_ts - float(getattr(self, "_pred_table_cols_cache_ts", 0.0))) < float(
                        getattr(self, "_pred_table_cols_cache_ttl", 600.0))):
                return set(self._pred_table_cols_cache)

            close_after = False
            if conn is None:
                conn = self.engine.connect()
                close_after = True

            try:
                cols = set(self._get_table_columns(self.TABLE_PRED, conn=conn))
                self._pred_table_cols_cache = set(cols)
                self._pred_table_cols_cache_ts = now_ts
                return set(cols)
            finally:
                if close_after:
                    try:
                        conn.close()
                    except Exception:
                        pass
        except Exception:
            return set()

    def get_latest_published_rmse(self):
        """
        读取最近一次 published 的训练记录的 RMSE（作为线上模型 baseline 兜底）。
        返回 float 或 None。
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            # 兼容字段名：publish_action / publish / status 等，以你表实际字段为准
            # 优先用 publish_action='published'
            sql = """
                SELECT evaluation_rmse
                FROM t_training_records
                WHERE (publish_action='published' OR publish='published')
                  AND evaluation_rmse IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
            """
            cursor.execute(sql)
            row = cursor.fetchone()
            try:
                cursor.close()
                conn.close()
            except Exception:
                pass
            if not row:
                return None
            # row 可能是 tuple/dict
            if isinstance(row, dict):
                return row.get("evaluation_rmse")
            return row[0]
        except Exception as e:
            logger.warning(f"get_latest_published_rmse失败：{repr(e)}")
            return None

    def get_recent_seed_samples_for_training(self, workface_id: int, before_date: str, limit: int = 1500) -> pd.DataFrame:
        """
        训练阶段使用：为跨批历史统计/趋势计算提供“历史种子行”
        - 只取最必要字段：measurement_date, workface_id, x/y/z, distance_to_face, 目标列(若存在)
        - 自动根据表结构过滤列，避免 Unknown column（例如 gas_emission_q 不存在）
        """
        try:
            dt = pd.to_datetime(before_date, errors="coerce")
            if pd.isna(dt):
                return pd.DataFrame()
            with self.engine.connect() as conn:
                pred_cols = self._get_prediction_table_columns(conn=conn)
                # 必要字段（没有就无法构建group）
                base_cols = ["measurement_date", "workface_id", "x_coord", "y_coord", "z_coord", "distance_to_face"]
                # 目标列：只取表里真实存在的
                candidate_targets = ["drilling_cuttings_s", "gas_emission_velocity_q", "gas_emission_q"]
                target_cols = [c for c in candidate_targets if c in pred_cols]
                select_cols = [c for c in base_cols if c in pred_cols] + target_cols
                if not select_cols:
                    return pd.DataFrame()
                sql = text(f"""
                    SELECT {", ".join(select_cols)}
                    FROM {self.TABLE_PRED}
                    WHERE workface_id = :wid
                      AND measurement_date < :dt
                    ORDER BY measurement_date DESC, id DESC
                    LIMIT :lim
                """)
                rows = conn.execute(sql, {"wid": int(workface_id), "dt": dt, "lim": int(limit)}).fetchall()
            if not rows:
                return pd.DataFrame(columns=select_cols)
            df = pd.DataFrame(rows, columns=select_cols)
            return df
        except Exception as e:
            logger.warning(f"get_recent_seed_samples_for_training失败：{repr(e)}", exc_info=True)
            return pd.DataFrame()

    def get_latest_feature_cache_by_group(self, workface_id: int, spatiotemporal_group: str,
                                          measurement_date: str) -> dict:
        """
        预测阶段使用：按 workface_id + spatiotemporal_group，取 measurement_date 之前最近一条增强特征（t_feature_cache）。
        关键原则：
          - 只返回白名单增强列（防止污染基础输入）
          - 只 SELECT 表里真实存在的列（防止 Unknown column）
          - 若白名单列缺失：明确 warning，提示“闭环未补齐/需迁移”，不掩盖问题
        """
        try:
            dt = pd.to_datetime(measurement_date, errors="coerce")
            if pd.isna(dt):
                return {}

            with self.engine.connect() as conn:
                # 不传 conn：避免旧版本签名/重复定义导致的关键字参数不兼容
                select_cols = self._get_feature_cache_select_cols()
                # getter 至少需要返回一组增强列；若为空则直接返回
                if not select_cols:
                    logger.warning("get_latest_feature_cache_by_group：feature_cache 可用列为空（请检查表结构/迁移）")
                    return {}

                # measurement_date 仅用于过滤/排序，不强制要求返回；但若schema有也可返回
                select_cols_sql = ", ".join([str(c) for c in select_cols if c != ""])

                sql = text(f"""
                    SELECT {select_cols_sql}
                    FROM {self.TABLE_FEATURE}
                    WHERE workface_id = :wid
                      AND spatiotemporal_group = :grp
                      AND measurement_date < :dt
                    ORDER BY measurement_date DESC, id DESC
                    LIMIT 1
                """)
                row = conn.execute(sql,
                                   {"wid": int(workface_id), "grp": str(spatiotemporal_group), "dt": dt}).fetchone()

            if not row:
                return {}

            d = {k: row[i] for i, k in enumerate(select_cols)}
            # 数值清洗：NaN -> 0（不改动非数值列）
            for k in list(d.keys()):
                v = d.get(k, None)
                if isinstance(v, float) and (v != v):
                    d[k] = 0.0
            return d
        except Exception as e:
            logger.warning(f"get_latest_feature_cache_by_group失败：{repr(e)}", exc_info=True)
            return {}

    def get_latest_feature_cache(self, workface_id: int, spatiotemporal_group: str, measurement_date: Optional[str] = None, is_training: bool = False) -> dict:
        """
        兼容 DataPreprocessor 的候选getter名称：get_latest_feature_cache
        - 若传 measurement_date：取该日期之前最近一条（推荐，避免用“未来增强”）
        - 若不传：取该 group 最新一条（可能引入时间穿越，谨慎使用；默认仍会 warning 提示）
        """
        try:
            if measurement_date:
                return self.get_latest_feature_cache_by_group(workface_id=workface_id, spatiotemporal_group=spatiotemporal_group, measurement_date=measurement_date)

            # 未给日期：退化为取最新一条（可能时间穿越，明确告警）
            if is_training:
                raise ValueError("训练阶段禁止 get_latest_feature_cache 不传 measurement_date（避免时间穿越）")

            # ---- [P0] 告警节流：同一 (wid, grp) 只提示一次，避免逐行刷屏 ----
            try:
                if not hasattr(self, "_warn_no_measurement_date_once"):
                    self._warn_no_measurement_date_once = set()
                key = (int(workface_id), str(spatiotemporal_group))
                if key not in self._warn_no_measurement_date_once:
                    self._warn_no_measurement_date_once.add(key)
                    logger.warning("预测阶段未传 measurement_date：将取该group最新增强特征（请尽量传日期）")
            except Exception:
                # 节流逻辑失败不影响主流程
                logger.warning("预测阶段未传 measurement_date：将取该group最新增强特征（请尽量传日期）")
            with self.engine.connect() as conn:
                select_cols = self._get_feature_cache_select_cols()
                if not select_cols:
                    return {}
                select_cols_sql = ", ".join([str(c) for c in select_cols if c != ""])
                sql = text(f"""
                    SELECT {select_cols_sql}
                    FROM {self.TABLE_FEATURE}
                    WHERE workface_id = :wid
                      AND spatiotemporal_group = :grp
                    ORDER BY measurement_date DESC, id DESC
                    LIMIT 1
                """)
                row = conn.execute(sql, {"wid": int(workface_id), "grp": str(spatiotemporal_group)}).fetchone()
            if not row:
                return {}
            d = {k: row[i] for i, k in enumerate(select_cols)}
            for k in list(d.keys()):
                v = d.get(k, None)
                if isinstance(v, float) and (v != v):
                    d[k] = 0.0
            return d
        except Exception as e:
            logger.warning(f"get_latest_feature_cache失败：{repr(e)}", exc_info=True)
            return {}

    def get_feature_cache_by_keys(self, workface_id: int, spatiotemporal_group: str, measurement_date: Optional[str] = None) -> dict:
        """
        兼容 DataPreprocessor 的候选getter名称：get_feature_cache_by_keys
        强约束：
          - 默认按 measurement_date 取“之前最近一条”（避免时间穿越）
          - 返回字段严格受白名单 + schema 过滤约束
          - 若未传 measurement_date：会warning并退化为 get_latest_feature_cache
        """
        try:
            if measurement_date:
                return self.get_latest_feature_cache_by_group(
                    workface_id=workface_id,
                    spatiotemporal_group=spatiotemporal_group,
                    measurement_date=measurement_date
                )
            logger.warning("get_feature_cache_by_keys 未传 measurement_date：将退化为 get_latest_feature_cache（可能时间穿越）")
            return self.get_latest_feature_cache(workface_id=workface_id, spatiotemporal_group=spatiotemporal_group, measurement_date=None)
        except Exception as e:
            logger.warning(f"get_feature_cache_by_keys失败：{repr(e)}", exc_info=True)
            return {}

    def get_recent_targets_by_group(
            self,
            workface_id: int,
            spatiotemporal_group: str,
            measurement_date: str,
            limit: int = 200,
            lookback_days: Optional[int] = None,
            dedupe_by_date: bool = True
        ) -> List[dict]:
        """
        训练阶段使用：按 workface_id + spatiotemporal_group，
        取 measurement_date 之前最近 N 条“真实目标值”（来自 t_prediction_parameters），
        但分组过滤以 t_feature_cache.spatiotemporal_group 为准（根治seed 0命中）。

        强约束：
          - 永不raise（失败返回[]）
          - 优先 JOIN t_feature_cache 命中；若 pred 表也存在 spatiotemporal_group，则回退直查 pred 表
        """
        try:
            dt = pd.to_datetime(measurement_date, errors="coerce")
            if pd.isna(dt):
                return []

            wid = int(workface_id) if workface_id is not None else None
            grp = str(spatiotemporal_group).strip()
            if wid is None or not grp:
                return []

            try:
                lim = int(limit)
                if lim <= 0:
                    return []
            except Exception:
                lim = 200
            # lookback_days：仅回捞最近N天（避免引入“远古分布”污染趋势）
            dt_lower = None
            try:
                if lookback_days is not None:
                    lb = int(lookback_days)
                    if lb > 0:
                        dt_lower = dt - timedelta(days=lb)
            except Exception:
                dt_lower = None

            # ---- 读取核心表字段集合（带缓存，避免频繁查 schema）----
            pred_cols = set()
            try:
                now_ts = time.time()
                if (getattr(self, "_pred_table_cols_cache", None) is not None and
                        (now_ts - float(getattr(self, "_pred_table_cols_cache_ts", 0.0))) < float(getattr(self, "_pred_table_cols_cache_ttl", 600.0))):
                    pred_cols = set(self._pred_table_cols_cache)
                else:
                    with self.engine.connect() as conn:
                        pred_cols = self._get_table_columns(self.TABLE_PRED, conn=conn)
                    self._pred_table_cols_cache = set(pred_cols)
                    self._pred_table_cols_cache_ts = now_ts
            except Exception:
                pred_cols = set()

            # ---- 目标列按“存在性”动态选择，避免 Unknown column ----
            target_cols = []
            for c in ["drilling_cuttings_s", "gas_emission_velocity_q", "gas_emission_q"]:
                if (not pred_cols) or (c in pred_cols):
                    # pred_cols 取不到时也先带上（由SQL执行决定），但一般能取到
                    target_cols.append(c)

            # 仅保留 pred 确认存在的列（更稳）
            if pred_cols:
                target_cols = [c for c in target_cols if c in pred_cols]

            # 至少需要 S、q 两个
            if "drilling_cuttings_s" not in target_cols or "gas_emission_velocity_q" not in target_cols:
                # 若库表异常缺列，直接不seed（不影响主流程）
                return []

            if dedupe_by_date:
                # 同一日期可能有多条（多孔/多深度），seed按日聚合：降低重复、抑制爆量
                select_sql = ", ".join(["p.measurement_date"] + [f"AVG(p.{c}) AS {c}" for c in target_cols])
                group_by_sql = " GROUP BY p.measurement_date "
            else:
                select_sql = ", ".join(["p.measurement_date"] + [f"p.{c}" for c in target_cols])
                group_by_sql = ""

            # 1) 优先：JOIN feature_cache 用 f.spatiotemporal_group 过滤（强保证seed命中）
            where_lb = ""
            if dt_lower is not None:
                where_lb = " AND p.measurement_date >= :dt_lower "

            sql_join = text(f"""
                SELECT {select_sql}
                FROM {self.TABLE_PRED} p
                INNER JOIN {self.TABLE_FEATURE} f
                    ON p.id = f.pred_id
                WHERE p.workface_id = :wid
                  AND f.spatiotemporal_group = :grp
                  AND p.measurement_date < :dt
                  {where_lb}
                {group_by_sql}
                ORDER BY p.measurement_date DESC
                LIMIT {int(lim)}
            """)

            # 2) 回退：若 pred 表本身也存在 spatiotemporal_group（兼容旧库）
            sql_fallback = None
            if ("spatiotemporal_group" in pred_cols):
                sql_fallback = text(f"""
                    SELECT {select_sql}
                    FROM {self.TABLE_PRED} p
                    WHERE p.workface_id = :wid
                      AND p.spatiotemporal_group = :grp
                      AND p.measurement_date < :dt
                      {where_lb}
                   {group_by_sql}
                    ORDER BY p.measurement_date DESC
                    LIMIT {int(lim)}
                """)

            rows = []
            with self.engine.connect() as conn:
                try:
                    params = {"wid": wid, "grp": grp, "dt": dt}
                    if dt_lower is not None:
                        params["dt_lower"] = dt_lower
                    rows = conn.execute(sql_join, params).fetchall()
                except Exception:
                    rows = []

                if (not rows) and sql_fallback is not None:
                    try:
                        params = {"wid": wid, "grp": grp, "dt": dt}
                        if dt_lower is not None:
                            params["dt_lower"] = dt_lower
                        rows = conn.execute(sql_fallback, params).fetchall()
                    except Exception:
                        rows = []

            if not rows:
                return []

            out = []
            for r in rows:
                d = {"measurement_date": r[0]}
                # r[1:] 对应 target_cols
                for i, c in enumerate(target_cols, start=1):
                    d[c] = r[i] if i < len(r) else None
                out.append(d)
            return out

        except Exception as e:
            logger.warning(f"get_recent_targets_by_group失败：{repr(e)}", exc_info=True)
            return []

    def get_recent_targets_by_coord_hash(
            self,
            workface_id: int,
            coord_hash: str,
            measurement_date: str,
            limit: int = 200
        ) -> List[dict]:
        """
        训练阶段回退：按 workface_id + coord_hash 取 measurement_date 之前最近 N 条真实目标值（q/S）。
        目的：当 spatiotemporal_group（尤其含 distance bucket）规则变化/不稳定导致 seed 0 命中时，用更粗粒度接续历史。

        强约束：永不raise，失败返回[]
        """
        try:
            dt = pd.to_datetime(measurement_date, errors="coerce")
            if pd.isna(dt):
                return []
            wid = int(workface_id) if workface_id is not None else None
            ch = str(coord_hash).strip()
            if wid is None or not ch:
                return []
            try:
                lim = int(limit)
                if lim <= 0:
                    return []
            except Exception:
                lim = 200

            # schema 探测：目标列按“存在性”动态选择
            pred_cols = set()
            try:
                now_ts = time.time()
                if (getattr(self, "_pred_table_cols_cache", None) is not None and
                        (now_ts - float(getattr(self, "_pred_table_cols_cache_ts", 0.0))) < float(getattr(self, "_pred_table_cols_cache_ttl", 600.0))):
                    pred_cols = set(self._pred_table_cols_cache)
                else:
                    with self.engine.connect() as conn:
                        pred_cols = self._get_table_columns(self.TABLE_PRED, conn=conn)
                    self._pred_table_cols_cache = set(pred_cols)
                    self._pred_table_cols_cache_ts = now_ts
            except Exception:
                pred_cols = set()

            target_cols = [c for c in ["drilling_cuttings_s", "gas_emission_velocity_q", "gas_emission_q"] if (not pred_cols) or (c in pred_cols)]
            if pred_cols:
                target_cols = [c for c in target_cols if c in pred_cols]
            if "drilling_cuttings_s" not in target_cols or "gas_emission_velocity_q" not in target_cols:
                return []

            select_sql = ", ".join(["p.measurement_date"] + [f"p.{c}" for c in target_cols])

            # 1) 优先 JOIN feature_cache 用 f.coord_hash 过滤（更稳：feature_cache 一定有coord_hash）
            sql_join = text(f"""
                SELECT {select_sql}
                FROM {self.TABLE_PRED} p
                INNER JOIN {self.TABLE_FEATURE} f
                    ON p.id = f.pred_id
                WHERE p.workface_id = :wid
                  AND f.coord_hash = :ch
                  AND p.measurement_date < :dt
                ORDER BY p.measurement_date DESC, p.id DESC
                LIMIT {int(lim)}
            """)

            # 2) 回退：若 pred 表本身也存在 coord_hash（兼容旧库/旧写入）
            sql_fallback = None
            if ("coord_hash" in pred_cols):
                sql_fallback = text(f"""
                    SELECT {select_sql}
                    FROM {self.TABLE_PRED} p
                    WHERE p.workface_id = :wid
                      AND p.coord_hash = :ch
                      AND p.measurement_date < :dt
                    ORDER BY p.measurement_date DESC, p.id DESC
                    LIMIT {int(lim)}
                """)

            rows = []
            with self.engine.connect() as conn:
                try:
                    rows = conn.execute(sql_join, {"wid": wid, "ch": ch, "dt": dt}).fetchall()
                except Exception:
                    rows = []
                if (not rows) and sql_fallback is not None:
                    try:
                        rows = conn.execute(sql_fallback, {"wid": wid, "ch": ch, "dt": dt}).fetchall()
                    except Exception:
                        rows = []

            if not rows:
                return []
            out = []
            for r in rows:
                d = {"measurement_date": r[0]}
                for i, c in enumerate(target_cols, start=1):
                    d[c] = r[i] if i < len(r) else None
                out.append(d)
            return out
        except Exception as e:
            logger.warning(f"get_recent_targets_by_coord_hash失败：{repr(e)}", exc_info=True)
            return []

    def get_recent_targets_by_workface(
            self,
            workface_id: int,
            measurement_date: str,
            limit: int = 200
        ) -> List[dict]:
        """
        训练阶段回退：仅按 workface_id 取 measurement_date 之前最近 N 条真实目标值（q/S）。
        目的：在 group/coord 维度都无法命中时，仍能提供“跨批历史种子”，避免历史均值/趋势全为0。

        强约束：永不raise，失败返回[]
        """
        try:
            dt = pd.to_datetime(measurement_date, errors="coerce")
            if pd.isna(dt):
                return []
            wid = int(workface_id) if workface_id is not None else None
            if wid is None:
                return []
            try:
                lim = int(limit)
                if lim <= 0:
                    return []
            except Exception:
                lim = 200

            pred_cols = set()
            try:
                now_ts = time.time()
                if (getattr(self, "_pred_table_cols_cache", None) is not None and
                        (now_ts - float(getattr(self, "_pred_table_cols_cache_ts", 0.0))) < float(getattr(self, "_pred_table_cols_cache_ttl", 600.0))):
                    pred_cols = set(self._pred_table_cols_cache)
                else:
                    with self.engine.connect() as conn:
                        pred_cols = self._get_table_columns(self.TABLE_PRED, conn=conn)
                    self._pred_table_cols_cache = set(pred_cols)
                    self._pred_table_cols_cache_ts = now_ts
            except Exception:
                pred_cols = set()

            target_cols = [c for c in ["drilling_cuttings_s", "gas_emission_velocity_q", "gas_emission_q"] if (not pred_cols) or (c in pred_cols)]
            if pred_cols:
                target_cols = [c for c in target_cols if c in pred_cols]
            if "drilling_cuttings_s" not in target_cols or "gas_emission_velocity_q" not in target_cols:
                return []

            select_sql = ", ".join(["p.measurement_date"] + [f"p.{c}" for c in target_cols])
            sql = text(f"""
                SELECT {select_sql}
                FROM {self.TABLE_PRED} p
                WHERE p.workface_id = :wid
                  AND p.measurement_date < :dt
                ORDER BY p.measurement_date DESC, p.id DESC
                LIMIT {int(lim)}
            """)

            rows = []
            with self.engine.connect() as conn:
                try:
                    rows = conn.execute(sql, {"wid": wid, "dt": dt}).fetchall()
                except Exception:
                    rows = []

            if not rows:
                return []
            out = []
            for r in rows:
                d = {"measurement_date": r[0]}
                for i, c in enumerate(target_cols, start=1):
                    d[c] = r[i] if i < len(r) else None
                out.append(d)
            return out
        except Exception as e:
            logger.warning(f"get_recent_targets_by_workface失败：{repr(e)}", exc_info=True)
            return []

    # -------------------------------------------------------------------------
    # 时序锚点（anchor_date）查询：解决增量训练 days_since_start 漂移/归零问题
    # -------------------------------------------------------------------------
    def get_workface_anchor_date(self, workface_id: int, work_stage: Optional[str] = None) -> Optional[pd.Timestamp]:
        """
        获取某 workface_id（可选 work_stage）在数据库中的“最早 measurement_date”，作为稳定时序锚点。
        设计目的：
          - 数据不一定从回采开始采集，但系统一旦开始采集，就用“系统首次采集日”作为 day0；
          - 后续增量训练/预测都用同一锚点计算 days_since_start，避免每批次按 df.min() 重置为0导致漂移。
        """
        try:
            wid = int(workface_id)
        except Exception:
            return None

        # 简单缓存（10分钟），避免频繁打库
        now = time.time()
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
            self._anchor_cache_ts = {}
            self._anchor_cache_ttl = 600

        cache_key = (wid, str(work_stage) if work_stage is not None else "")
        ts = self._anchor_cache_ts.get(cache_key, 0.0)
        if cache_key in self._anchor_cache and (now - ts) < self._anchor_cache_ttl:
            return self._anchor_cache.get(cache_key)

        # 先从核心表找最早 measurement_date（最权威）
        stage_clause = ""
        params = {"wid": wid}
        if work_stage is not None and str(work_stage).strip() != "":
            stage_clause = " AND work_stage = :ws "
            params["ws"] = str(work_stage).strip()

        sql_min_pred = text(f"""
            SELECT MIN(measurement_date) AS min_dt
            FROM {self.TABLE_PRED}
            WHERE workface_id = :wid
            {stage_clause}
        """)

        anchor = None
        with self.engine.connect() as conn:
            row = conn.execute(sql_min_pred, params).fetchone()
            if row and row[0]:
                anchor = pd.to_datetime(row[0], errors="coerce")

            # 若核心表没有，再尝试从 feature_cache 找（兜底）
            if anchor is None or pd.isna(anchor):
                sql_min_feat = text(f"""
                    SELECT MIN(measurement_date) AS min_dt
                    FROM {self.TABLE_FEATURE}
                    WHERE workface_id = :wid
                """)
                row2 = conn.execute(sql_min_feat, {"wid": wid}).fetchone()
                if row2 and row2[0]:
                    anchor = pd.to_datetime(row2[0], errors="coerce")

        if anchor is not None and not pd.isna(anchor):
            # 统一到“天”级，避免时分秒带来的负偏差
            try:
                anchor = pd.to_datetime(anchor).normalize()
            except Exception:
                pass
        else:
            anchor = None

        self._anchor_cache[cache_key] = anchor
        self._anchor_cache_ts[cache_key] = now
        return anchor

    # -------------------- t_feature_cache 返回字段白名单（只允许增强特征） --------------------
    # 说明：
    # 1) 只允许“增强列”从缓存表回填，避免污染基础输入列
    # 2) 本白名单会再叠加“真实schema过滤”，只select表里存在的列
    FEATURE_CACHE_WHITELIST: List[str] = [
        "coord_hash",
        "spatiotemporal_group",
        "distance_to_face_bucket",
        "days_since_start",
        "days_in_workface",
        "distance_time_interaction",
        "drilling_cuttings_s_historical_mean",
        "gas_emission_velocity_q_historical_mean",
        "gas_emission_q_historical_mean",
        "drilling_cuttings_s_trend",
        "gas_emission_velocity_q_trend",
        "gas_emission_q_trend",
        # ---- 新增增强列（必须闭环：生成→写库→读库→回填）----
        "gas_emission_velocity_q_rolling_mean",
        "gas_emission_velocity_q_hist_fused",
        # ---------------------------------------------------
        "advance_rate",
        "measurement_date",
    ]

    def _get_feature_table_columns(self) -> Set[str]:
        '''
        获取 t_feature_cache 真实字段集合（带缓存，避免频繁查 schema）
        强约束：失败时返回空集合（调用方应做好兜底）
        '''
        logger.debug(f"[诊断] _get_feature_table_columns 签名=()，函数对象={self._get_feature_table_columns}")
        try:
            now_ts = time.time()
            if (getattr(self, "_feature_table_cols_cache", None) is not None and
                    (now_ts - float(getattr(self, "_feature_table_cols_cache_ts", 0.0))) < float(getattr(self, "_feature_table_cols_cache_ttl", 600.0))):
                return set(self._feature_table_cols_cache)

            with self.engine.connect() as conn:
                cols = set(self._get_table_columns(self.TABLE_FEATURE, conn=conn))
            self._feature_table_cols_cache = set(cols)
            self._feature_table_cols_cache_ts = now_ts
            return set(cols)
        except Exception:
            return set()

    def build_join_column_lists(self,
                                needed_cols: Optional[List[str]] = None,
                                include_targets: bool = True) -> (List[str], List[str]):
        '''
        统一列清单生成器：用于任何 “pred + feature_cache” JOIN 取数场景
        返回：
          - p_cols：从 t_prediction_parameters 读取的字段名列表（不含别名）
          - f_cols：从 t_feature_cache 读取的字段名列表（不含别名，拼 SQL 时统一 `f.{c} AS {c}`）

        设计目标：
        1) 避免重复列名（增强特征只允许来自 feature_cache）
        2) 避免 Unknown column（最终用 DB schema 白名单过滤）
        3) 永久禁止 gas_emission_q 及其派生列进入任何 JOIN 取数（与你现阶段“只预测S、q”一致）
        '''
        # 1) 默认索引字段（保证评估/窗口取数稳定）
        p_index_cols = [
            "id", "working_face", "workface_id", "work_stage",
            "roadway", "roadway_id", "measurement_date",
            "borehole_id", "x_coord", "y_coord", "z_coord",
            "distance_to_face", "distance_from_entrance"
        ]

        # 2) 默认 feature_cache 增强列（这些列允许来自 f）
        f_cols_default = [
            "coord_hash",
            "spatiotemporal_group",
            "distance_to_face_bucket",
            "days_since_start",
            "days_in_workface",
            "distance_time_interaction",
            "drilling_cuttings_s_trend",
            "gas_emission_velocity_q_trend",
            "drilling_cuttings_s_historical_mean",
            "gas_emission_velocity_q_historical_mean",
            "advance_rate",
        ]

        # 3) 默认目标列（仅 S、q；gas_emission_q 永久禁用）
        target_cols = ["drilling_cuttings_s", "gas_emission_velocity_q"] if include_targets else []

        # 4) 需要的列集合（清洗 + 去重）
        need = []
        if needed_cols:
            need.extend(self._normalize_col_list(needed_cols))
        need.extend([c for c in self._normalize_col_list(p_index_cols) if c not in need])
        need.extend([c for c in self._normalize_col_list(target_cols) if c not in need])

        # 永久禁止：gas_emission_q 及其派生列
        need = [c for c in need if not (c == "gas_emission_q" or c.startswith("gas_emission_q_"))]

        # 永久禁止：任何 advance_rate 变体从 pred 表读取（推进速率统一来自 feature_cache）
        banned_pred_exact = {
            "advance_rate", "advance_rate_x", "advance_rate_y", "advance_rate_mining"
        }

        pred_cols = self._get_prediction_table_columns()
        feat_cols = self._get_feature_table_columns()

        # 5) 按 schema 划分 p/f 列（若 schema 取不到，采用“保守策略”：只用默认列）
        p_out, f_out = [], []
        if pred_cols or feat_cols:
            for c in need:
                # f 优先：只要 feature_cache 真实存在该列，就让它来自 f
                if feat_cols and c in feat_cols:
                    if c not in f_out:
                        f_out.append(c)
                    continue
                # 否则尝试来自 p（必须存在）
                if pred_cols and c in pred_cols and c not in banned_pred_exact:
                    if c not in p_out:
                        p_out.append(c)
        else:
            # schema 取不到时：只返回最小可用集合（避免拼错列名）
            p_out = self._normalize_col_list(p_index_cols + target_cols)
            f_out = self._normalize_col_list(f_cols_default)

        # 6) 将默认增强列补入 f（如果存在于 feature_cache）
        if feat_cols:
            for c in self._normalize_col_list(f_cols_default):
                if c in feat_cols and c not in f_out:
                    f_out.append(c)

        # 7) 最终保证：f 列不与 p 列重复；且 p 不含任何 banned_pred
        f_set = set(f_out)
        p_out = [c for c in p_out if (c not in f_set and c not in banned_pred_exact)]
        p_out = self._normalize_col_list(p_out)
        f_out = self._normalize_col_list(f_out)

        return p_out, f_out

    def _get_feature_cache_select_cols(self, needed_cols: Optional[List[str]] = None) -> List[str]:
        """
        生成 t_feature_cache 的 SELECT 列清单：
        - 先走白名单（只允许增强列）
        - 再用真实schema过滤（只select存在的列）
        - 如果 needed_cols 传入：在白名单基础上再裁剪（仍要schema过滤）
        - 不掩盖问题：白名单列缺失会 warning 提示
        """
        try:
            schema_cols = self._get_feature_table_columns()
            # 1) 白名单
            wl = list(self.FEATURE_CACHE_WHITELIST)
            if needed_cols:
                try:
                    need_set = {str(c).strip() for c in needed_cols if isinstance(c, str) and str(c).strip()}
                except Exception:
                    need_set = set()
                if need_set:
                    wl = [c for c in wl if c in need_set]

            # 2) schema 过滤
            if schema_cols:
                missing = [c for c in wl if c not in schema_cols]
                if missing:
                    logger.warning(
                        f"t_feature_cache 缺少 {len(missing)} 个白名单字段（需做闭环补齐/迁移）：{missing}"
                    )
                wl = [c for c in wl if c in schema_cols]

            # 3) 至少返回一个可用列（否则上层直接返回{}）
            return wl
        except Exception as e:
            logger.warning(f"_get_feature_cache_select_cols失败（将退回白名单基础版）：{repr(e)}", exc_info=True)
            return [c for c in self.FEATURE_CACHE_WHITELIST if c != ""]

    def _get_table_columns(self, table_name: str, conn: Optional[Connection] = None) -> Set[str]:
        """
        获取任意表字段集合（不缓存）
        """
        close_after = False
        if conn is None:
            conn = self._get_connection()
            close_after = True
        try:
            rows = conn.execute(text("""
                SELECT COLUMN_NAME
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :table_name
            """), {"table_name": table_name}).fetchall()
            return {r[0] for r in rows} if rows else set()
        except Exception as e:
            logger.warning(f"读取表字段失败（{table_name}）：{str(e)}")
            return set()
        finally:
            if close_after:
                try:
                    conn.close()
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # 分表写入：训练样本
    # -------------------------------------------------------------------------
    def save_training_data(
        self,
        df: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None,
        conn: Optional[Connection] = None,
        trans: Any = None,
        custom_create_time: Optional[str] = None
    ) -> int:
        """
        保存训练数据（分表写入：核心表 + 分源专表 + 特征缓存专表）

        :param df: DataFrame
        :param column_mapping: 字段映射（df列名 -> db列名）
        :param conn: 外部连接（可选）
        :param trans: 外部事务（可选，与conn配套）
        :param custom_create_time: 指定 create_time（用于回滚定位/审计）
        :return: 成功写入核心表的条数
        """
        if df is None or df.empty:
            logger.warning("无数据需保存到数据库（df为空）")
            return 0

        mapped_df = df.copy()

        # Step 1: 字段映射
        if column_mapping:
            valid_cols = [c for c in mapped_df.columns if c in column_mapping]
            mapped_df = mapped_df[valid_cols].rename(columns=column_mapping)

        # Step 2: 自定义 create_time
        if custom_create_time:
            mapped_df["create_time"] = custom_create_time
        # Step 2.1: 数据库兼容清洗（MySQL不接受NaN/NaT，需统一转为None）
        # 说明：pymysql遇到nan会报：nan can not be used with MySQL
        try:
            mapped_df = mapped_df.where(pd.notna(mapped_df), None)
        except Exception as e:
            logger.warning(f"NaN清洗失败：{str(e)}（将继续尝试写库，可能失败）")
        records_all = mapped_df.to_dict(orient="records")
        if not records_all:
            logger.warning("数据转换为dict失败（无有效记录）")
            return 0

        # Step 3: 三类字段集合（写入策略）
        core_cols = [
            # 最少索引
            "working_face", "workface_id", "work_stage", "roadway", "roadway_id",
            # 空间/时间/距离/钻孔
            "x_coord", "y_coord", "z_coord", "distance_from_entrance",
            "borehole_id", "drilling_depth", "measurement_date",
            "distance_to_face", "face_advance_distance",
            # 基础特征
            "coal_thickness", "fault_influence_strength", "regional_measure_strength",
            # 目标值
            "drilling_cuttings_s", "gas_emission_velocity_q",
            # 审计
            "create_time",
        ]

        source_cols = [
            "tunneling_speed", "roadway_length", "initial_gas_emission_strength",
            "roadway_cross_section", "coal_density", "original_gas_content", "residual_gas_content",
            "create_time",
        ]

        feature_cols = [
            "coord_hash", "spatiotemporal_group", "distance_to_face_bucket",
            "days_since_start", "days_in_workface", "distance_time_interaction",
            "gas_emission_q_historical_mean", "drilling_cuttings_s_historical_mean", "gas_emission_velocity_q_historical_mean",
            "gas_emission_velocity_q_rolling_mean",
            "gas_emission_velocity_q_hist_fused",
            "gas_emission_q_trend", "drilling_cuttings_s_trend", "gas_emission_velocity_q_trend",
            "advance_rate",
            "create_time",
        ]

        # Step 4: 表字段交集过滤（防字段漂移）
        try:
            pred_cols = self._get_prediction_table_columns(conn=conn)
            src_cols = self._get_table_columns(self.TABLE_SOURCE, conn=conn)
            feat_cols = self._get_table_columns(self.TABLE_FEATURE, conn=conn)

            core_cols = [c for c in core_cols if c in pred_cols] if pred_cols else core_cols
            source_cols = [c for c in source_cols if c in src_cols] if src_cols else source_cols
            feature_cols = [c for c in feature_cols if c in feat_cols] if feat_cols else feature_cols
        except Exception as e:
            logger.warning(f"分表字段交集过滤失败：{str(e)}（将继续按预设字段写入，可能存在风险）")

        if not core_cols:
            logger.warning("核心写入列为空，无法写入训练数据")
            return 0

        # Step 5: 构造 SQL（核心表按行插入，以获得 pred_id）
        core_insert_sql = text(f"""
            INSERT INTO {self.TABLE_PRED} ({', '.join(core_cols)})
            VALUES ({', '.join([':' + c for c in core_cols])})
        """)

        # 分源专表：固定包含 pred_id + 冗余索引字段
        src_fields = [
            "pred_id", "working_face", "workface_id", "work_stage", "roadway", "roadway_id",
            "measurement_date", "x_coord", "y_coord", "z_coord", "drilling_depth", "distance_to_face",
        ] + [c for c in source_cols if c != "create_time"] + (["create_time"] if "create_time" in source_cols else [])

        src_insert_sql = text(f"""
            INSERT INTO {self.TABLE_SOURCE} ({', '.join(src_fields)})
            VALUES ({', '.join([':' + c for c in src_fields])})
        """)

        # 特征缓存表：固定包含 pred_id + 冗余索引字段
        feat_fields = [
            "pred_id", "working_face", "workface_id", "work_stage", "measurement_date",
            "x_coord", "y_coord", "z_coord", "distance_to_face",
        ] + [c for c in feature_cols if c != "create_time"] + (["create_time"] if "create_time" in feature_cols else [])

        feat_insert_sql = text(f"""
            INSERT INTO {self.TABLE_FEATURE} ({', '.join(feat_fields)})
            VALUES ({', '.join([':' + c for c in feat_fields])})
        """)

        def _do_insert(target_conn: Connection) -> int:
            saved = 0
            for row in records_all:
                # 5.1 核心表写入
                core_payload = {c: row.get(c) for c in core_cols}
                res = target_conn.execute(core_insert_sql, core_payload)
                # 兼容：部分环境下 lastrowid 可能取不到，使用 LAST_INSERT_ID() 兜底
                pred_id = getattr(res, "lastrowid", None)
                if pred_id is None:
                    try:
                        pred_id = target_conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
                    except Exception:
                        pred_id = None
                saved += 1

                # 5.2 公共冗余字段
                common = {
                    "pred_id": pred_id,
                    "working_face": row.get("working_face"),
                    "workface_id": row.get("workface_id"),
                    "work_stage": row.get("work_stage"),
                    "roadway": row.get("roadway"),
                    "roadway_id": row.get("roadway_id"),
                    "measurement_date": row.get("measurement_date"),
                    "x_coord": row.get("x_coord"),
                    "y_coord": row.get("y_coord"),
                    "z_coord": row.get("z_coord"),
                    "drilling_depth": row.get("drilling_depth"),
                    "distance_to_face": row.get("distance_to_face"),
                }

                # 5.3 分源专表：任一分源字段有值才写
                # 仅当存在“非None且非NaN”的字段值时才写入分源专表
                def _has_real_value(v):
                    if v is None:
                        return False
                    try:
                        # 兼容float('nan')：nan != nan
                        return not (isinstance(v, float) and v != v)
                    except Exception:
                        return True
                if any(_has_real_value(row.get(c)) for c in source_cols if c != "create_time"):
                    src_payload = {k: None for k in src_fields}
                    # 填公共字段
                    for k in src_payload.keys():
                        if k in common:
                            src_payload[k] = common.get(k)
                    # 填分源字段
                    for c in source_cols:
                        if c in src_payload:
                            src_payload[c] = row.get(c)
                    # ---- 诊断：确保pred_id键存在（缺键会触发 KeyError('pred_id')）----
                    if "pred_id" not in src_payload:
                        logger.error(f"[诊断] src_payload 缺少 pred_id，src_fields={src_fields}")
                        logger.error(f"[诊断] src_payload keys={list(src_payload.keys())}")
                        raise KeyError("pred_id")
                    # 写库前清洗 NaN/NaT -> None（避免MySQL写入nan/NaT导致异常）
                    for kk, vv in list(src_payload.items()):
                        try:
                            if vv is None:
                                continue
                            if isinstance(vv, float) and vv != vv:
                                src_payload[kk] = None
                                continue
                            if pd.isna(vv):
                                src_payload[kk] = None
                        except Exception:
                            # 保守处理：不动
                            pass
                    try:
                        target_conn.execute(src_insert_sql, src_payload)
                    except Exception as e:
                        logger.error(f"[诊断] 插入 {self.TABLE_SOURCE} 失败：{repr(e)}", exc_info=True)
                        logger.error(f"[诊断] src_fields={src_fields}")
                        logger.error(f"[诊断] src_payload keys={list(src_payload.keys())}")
                        logger.error(f"[诊断] src_payload pred_id={src_payload.get('pred_id')}")
                        raise

                # 5.4 特征缓存表：任一增强字段有值才写
                if any(_has_real_value(row.get(c)) for c in feature_cols if c != "create_time"):
                    feat_payload = {k: None for k in feat_fields}
                    for k in feat_payload.keys():
                        if k in common:
                            feat_payload[k] = common.get(k)
                    for c in feature_cols:
                        if c in feat_payload:
                            feat_payload[c] = row.get(c)

                    if "pred_id" not in feat_payload:
                        logger.error(f"[诊断] feat_payload 缺少 pred_id，feat_fields={feat_fields}")
                        logger.error(f"[诊断] feat_payload keys={list(feat_payload.keys())}")
                        raise KeyError("pred_id")
                    # ---- 关键修复：写库前强制清洗NaN/NaT为None，避免pymysql报 nan can not be used with MySQL ----
                    for kk, vv in list(feat_payload.items()):
                        # float NaN：nan != nan
                        if isinstance(vv, float) and vv != vv:
                            feat_payload[kk] = None
                        else:
                            try:
                                # pandas 的 NA/NaT
                                if pd.isna(vv):
                                    feat_payload[kk] = None
                            except Exception:
                                pass
                    try:
                        target_conn.execute(feat_insert_sql, feat_payload)
                    except Exception as e:
                        logger.error(f"[诊断] 插入 {self.TABLE_FEATURE} 失败：{repr(e)}", exc_info=True)
                        logger.error(f"[诊断] feat_fields={feat_fields}")
                        logger.error(f"[诊断] feat_payload keys={list(feat_payload.keys())}")
                        logger.error(f"[诊断] feat_payload pred_id={feat_payload.get('pred_id')}")
                        raise
            return saved

        # Step 6: 执行（优先外部事务）
        try:
            if conn is not None and trans is not None:
                saved_count = _do_insert(conn)
                logger.debug(f"外部事务中分表插入数据：{saved_count}条（未提交）")
                return saved_count

            inner_conn = self._get_connection()
            inner_trans = inner_conn.begin()
            try:
                saved_count = _do_insert(inner_conn)
                inner_trans.commit()
                logger.info(f"内部事务中分表保存数据：{saved_count}条（已提交）")
                return saved_count
            except Exception as e:
                inner_trans.rollback()
                logger.error(f"内部事务分表插入失败：{str(e)}，已回滚", exc_info=True)
                return 0
            finally:
                try:
                    inner_conn.close()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"保存训练数据失败（分表写入）：{str(e)}", exc_info=True)
            return 0

    # -------------------------------------------------------------------------
    # 训练记录写入
    # -------------------------------------------------------------------------
    def insert_training_record(self, record: Dict[str, Any]) -> Optional[int]:
        """
        插入训练记录到 t_training_records
        """
        if not record:
            logger.warning("训练记录为空，不插入")
            return None

        payload = {
            "sample_count": record.get("sample_count", 0),
            "total_samples": record.get("total_samples", 0),
            "train_mode": record.get("train_mode", "未知"),
            "status": record.get("status", "unknown"),
            "message": record.get("message", ""),
            "duration": record.get("duration", 0.0),
            "train_time": record.get("train_time") or None,
        }

        conn = self._get_connection()
        trans = conn.begin()
        try:
            res = conn.execute(text(f"""
                INSERT INTO {self.TABLE_TRAIN}
                (sample_count, total_samples, train_mode, status, message, duration, train_time)
                VALUES (:sample_count, :total_samples, :train_mode, :status, :message, :duration, :train_time)
            """), payload)
            trans.commit()
            rid = getattr(res, "lastrowid", None)
            logger.debug(f"训练记录插入成功，ID={rid}")
            return rid
        except Exception as e:
            trans.rollback()
            logger.error(f"训练记录插入失败：{str(e)}", exc_info=True)
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 断层查询（FaultCalculator 依赖）
    # -------------------------------------------------------------------------
    def get_faults_by_workface(self, workface_id: int) -> List[Dict[str, Any]]:
        """
        查询某工作面下所有断层（t_geo_fault）
        """
        conn = self._get_connection()
        try:
            res = conn.execute(text(f"""
                SELECT id, name, length, azimuth, inclination,
                       fault_height, fault_type, influence_scope
                FROM {self.TABLE_FAULT}
                WHERE workface_id = :workface_id
            """), {"workface_id": workface_id})
            return [dict(r._mapping) for r in res.fetchall()]
        except Exception as e:
            logger.error(f"查询断层失败（workface_id={workface_id}）：{str(e)}", exc_info=True)
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_fault_points(self, geofault_id: int) -> List[Dict[str, Any]]:
        """
        查询断层组成点（t_coal_point）
        """
        conn = self._get_connection()
        try:
            res = conn.execute(text(f"""
                SELECT floor_coordinate_x, floor_coordinate_y, floor_coordinate_z
                FROM {self.TABLE_FAULT_POINT}
                WHERE geofault_id = :geofault_id
                ORDER BY point_order ASC
            """), {"geofault_id": geofault_id})
            return [dict(r._mapping) for r in res.fetchall()]
        except Exception as e:
            logger.error(f"查询断层点失败（geofault_id={geofault_id}）：{str(e)}", exc_info=True)
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def fetch_recent_training_window_with_features(
            self,
            workface_ids=None,
            max_date=None,
            lookback_days: int = 14,
            limit: int = 2000,
            conn=None,
            # --------- 兼容旧调用参数（避免别处旧签名调用崩）---------
            workface_id: int = None,
            work_stage: str = None,
            needed_cols=None
    ) -> pd.DataFrame:
        """
        增量训练窗口取数：从 t_prediction_parameters 回捞最近 lookback_days 天数据，
        并 LEFT JOIN t_feature_cache 取增强列，返回合并后的 DataFrame。

        强保证：
          1) 永不 raise：任何异常返回空 DataFrame
          2) lookback 必然生效：窗口范围 [max_date - lookback_days, max_date]
          3) 传入 conn（同事务）时可读取未提交数据（窗口包含当前批次）
          4) 100% 不引用 gas_emission_q（无论 pred / feature / needed_cols）
        """
        import pandas as pd
        from datetime import timedelta
        from sqlalchemy import text, bindparam

        try:
            # -------------------------
            # 0) 参数防御 + 统一口径
            # -------------------------
            try:
                lookback_days = int(lookback_days)
            except Exception:
                lookback_days = 14
            if lookback_days <= 0:
                lookback_days = 14

            try:
                limit = int(limit)
            except Exception:
                limit = 2000
            if limit <= 0:
                return pd.DataFrame()

            # 兼容旧签名：workface_id -> workface_ids
            if (not workface_ids) and (workface_id is not None):
                try:
                    workface_ids = [int(float(workface_id))]
                except Exception:
                    workface_ids = None

            # workface_ids 清洗
            if workface_ids:
                try:
                    _tmp = []
                    for x in workface_ids:
                        try:
                            _tmp.append(int(float(x)))
                        except Exception:
                            continue
                    workface_ids = sorted(set(_tmp)) if _tmp else None
                except Exception:
                    workface_ids = None

            # work_stage 清洗（可为空）
            if work_stage is not None:
                try:
                    work_stage = str(work_stage).strip()
                    if work_stage == "":
                        work_stage = None
                except Exception:
                    work_stage = None

            close_after = False
            if conn is None:
                conn = self.engine.connect()
                close_after = True

            # -------------------------
            # 1) 计算窗口上界 dt_max（优先使用传入 max_date）
            # -------------------------
            dt_max = None
            if max_date:
                try:
                    dt_max = pd.to_datetime(max_date, errors="coerce")
                except Exception:
                    dt_max = None

            if dt_max is None or pd.isna(dt_max):
                # 若没传 max_date，则用同事务 conn 从库里取 MAX(measurement_date)
                where_parts = []
                params = {}

                if workface_ids:
                    where_parts.append("p.workface_id IN :wids")
                    params["wids"] = workface_ids

                if work_stage is not None:
                    where_parts.append("p.work_stage = :ws")
                    params["ws"] = work_stage

                where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
                sql_max = text(f"SELECT MAX(p.measurement_date) AS max_dt FROM {self.TABLE_PRED} p {where_sql}")
                if "wids" in params:
                    sql_max = sql_max.bindparams(bindparam("wids", expanding=True))

                row = conn.execute(sql_max, params).fetchone()
                if row and row[0] is not None:
                    dt_max = pd.to_datetime(row[0], errors="coerce")

            if dt_max is None or pd.isna(dt_max):
                return pd.DataFrame()

            try:
                dt_max = pd.to_datetime(dt_max).normalize()
            except Exception:
                pass

            # -------------------------
            # 2) 计算窗口下界 dt_min（lookback 必然生效）
            # -------------------------
            dt_min = dt_max - timedelta(days=int(lookback_days))

            # -------------------------
            # 3) 统一列清单生成器（按 schema 白名单生成 p/f 两表列清单）
            # -------------------------
            p_cols, f_cols = self.build_join_column_lists(
                needed_cols=needed_cols,
                include_targets=True
            )

            # 组装 SELECT 列（feature 列全部显式 AS，避免重复列名）
            select_cols = [f"p.{c}" for c in p_cols] + [f"f.{c} AS {c}" for c in f_cols]

            # -------------------------
            # 4) 最终 SQL（无 gas_emission_q）
            # -------------------------
            where_parts = [
                "p.measurement_date >= :dt_min",
                "p.measurement_date <= :dt_max",
            ]
            params = {
                "dt_min": str(pd.to_datetime(dt_min).date()),
                "dt_max": str(pd.to_datetime(dt_max).date()),
            }

            if workface_ids:
                where_parts.append("p.workface_id IN :wids")
                params["wids"] = workface_ids

            if work_stage is not None:
                where_parts.append("p.work_stage = :ws")
                params["ws"] = work_stage

            where_sql = "WHERE " + " AND ".join(where_parts)

            sql = text(f"""
                SELECT
                    {", ".join(select_cols)}
                FROM {self.TABLE_PRED} p
                LEFT JOIN {self.TABLE_FEATURE} f
                  ON p.id = f.pred_id
                {where_sql}
                ORDER BY p.measurement_date DESC, p.id DESC
                LIMIT {int(limit)}
            """)

            if "wids" in params:
                sql = sql.bindparams(bindparam("wids", expanding=True))

            df = pd.read_sql(sql, conn, params=params)

            # measurement_date 统一为字符串（与你主流程一致，避免类型不一致）
            try:
                if "measurement_date" in df.columns:
                    df["measurement_date"] = df["measurement_date"].astype(str)
            except Exception:
                pass

            return df if df is not None else pd.DataFrame()

        except Exception as e:
            try:
                logger.warning(f"fetch_recent_training_window_with_features失败（已兜底返回空df）：{repr(e)}",
                               exc_info=True)
            except Exception:
                pass
            return pd.DataFrame()

        finally:
            try:
                if close_after and conn is not None:
                    conn.close()
            except Exception:
                pass

