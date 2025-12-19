"""
煤矿瓦斯风险预测系统 - 数据库工具模块
作用：基于SQLAlchemy 2.x实现数据库操作，支持：
  1. 自动初始化表结构（含断层相关表、动态字段）
  2. 批量插入训练数据（支持外部事务，用于train方法原子性控制）
  3. 断层数据查询（按工作面ID查断层、按断层ID查组成点）
  4. 训练记录插入（t_training_records表）
依赖：SQLAlchemy、pymysql、loguru、tenacity（重试机制）
"""
import configparser
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class DBUtils:
    """
    数据库操作工具类（兼容SQLAlchemy 2.x，线程安全）
    核心能力：表初始化、数据CRUD、断层数据查询
    """

    def __init__(self, config_path="config.ini"):
        """
        初始化DBUtils，读取数据库配置并创建SQLAlchemy引擎

        :param config_path: str，配置文件路径，默认"config.ini"
        :raises Exception: 数据库配置缺失时抛出异常
        """
        # Step 1: 读取数据库配置
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding="utf-8")
        try:
            self.db_config = {
                "host": self.config.get("Database", "host"),
                "port": self.config.get("Database", "port"),
                "user": self.config.get("Database", "user"),
                "password": self.config.get("Database", "password"),
                "db": self.config.get("Database", "db_name"),
                "charset": self.config.get("Database", "charset", fallback="utf8mb4")
            }
        except Exception as e:
            logger.error(f"数据库配置缺失：{str(e)}（检查[Database] section）")
            raise

        # Step 2: 创建SQLAlchemy引擎（pymysql驱动，连接池预检测）
        conn_str = (
            f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['db']}?charset={self.db_config['charset']}"
        )
        self.engine = create_engine(
            conn_str,
            pool_pre_ping=True,  # 连接池预检测，避免无效连接
            future=True,  # 启用SQLAlchemy 2.x特性
            pool_size=5,  # 连接池大小（生产环境可调整）
            max_overflow=10  # 最大溢出连接数
        )

        # Step 3: 加载动态字段配置（从[Features] section）
        self._load_dynamic_columns()

        # Step 4: 初始化表结构（不存在则创建）
        self._init_tables()
        logger.debug("DBUtils初始化完成，数据库引擎已创建")

    def reload_config(self, config_path=None):
        """
        动态重载数据库配置（谨慎使用，会重建数据库连接）

        :param config_path: str，新配置文件路径
        :return: bool，重载是否成功
        """
        try:
            logger.warning("数据库配置重载：将重建数据库连接池")

            # 备份当前连接信息（用于日志）
            old_config = self.db_config.copy()

            # 重新读取配置
            if config_path:
                self.config.read(config_path, encoding="utf-8")

            # 重新获取数据库配置
            try:
                new_db_config = {
                    "host": self.config.get("Database", "host"),
                    "port": self.config.get("Database", "port"),
                    "user": self.config.get("Database", "user"),
                    "password": self.config.get("Database", "password"),
                    "db": self.config.get("Database", "db_name"),
                    "charset": self.config.get("Database", "charset", fallback="utf8mb4")
                }
            except Exception as e:
                logger.error(f"数据库配置读取失败：{str(e)}")
                return False

            # 检查配置是否真的发生变化
            config_changed = any(old_config.get(k) != new_db_config.get(k) for k in old_config)
            if not config_changed:
                logger.info("数据库配置未发生变化，跳过重载")
                return True

            # 重建数据库引擎
            conn_str = (
                f"mysql+pymysql://{new_db_config['user']}:{new_db_config['password']}@"
                f"{new_db_config['host']}:{new_db_config['port']}/{new_db_config['db']}?charset={new_db_config['charset']}"
            )

            # 先创建新引擎，再替换旧引擎（避免服务中断）
            new_engine = create_engine(
                conn_str,
                pool_pre_ping=True,
                future=True,
                pool_size=5,
                max_overflow=10
            )

            # 测试新连接
            try:
                with new_engine.connect() as test_conn:
                    test_conn.execute(text("SELECT 1"))
                logger.debug("新数据库连接测试成功")
            except Exception as e:
                logger.error(f"新数据库连接测试失败：{str(e)}")
                return False

            # 替换引擎
            old_engine = self.engine
            self.engine = new_engine
            self.db_config = new_db_config

            # 关闭旧连接池
            try:
                old_engine.dispose()
                logger.debug("旧数据库连接池已清理")
            except Exception as e:
                logger.warning(f"清理旧数据库连接池失败：{str(e)}")

            logger.info("数据库配置重载完成")
            return True

        except Exception as e:
            logger.error(f"数据库配置重载失败：{str(e)}", exc_info=True)
            return False

    def _load_dynamic_columns(self):
        """
        私有方法：从config.ini加载动态字段配置（[Features]的dynamic_db_columns）
        格式：字段名:数据类型;如"new_feature1:FLOAT;new_feature2:VARCHAR(100)"
        """
        self.dynamic_columns = {}
        try:
            dynamic_cols_str = self.config.get("Features", "dynamic_db_columns", fallback="")
            if dynamic_cols_str:
                for col_def in dynamic_cols_str.split(";"):
                    if ":" in col_def:
                        col_name, col_type = col_def.split(":", 1)
                        self.dynamic_columns[col_name.strip()] = col_type.strip()
            logger.debug(f"加载动态字段配置：{self.dynamic_columns}")
        except Exception as e:
            logger.error(f"加载动态字段配置失败：{str(e)}，使用空配置")
            self.dynamic_columns = {}


    def _get_connection(self):
        """
        上下文管理器：获取SQLAlchemy Connection对象（自动关闭连接）

        :yield: sqlalchemy.engine.Connection，数据库连接对象
        :raises SQLAlchemyError: 连接或执行错误时抛出
        """
        try:
            conn = self.engine.connect()
            logger.debug(f"数据库连接已创建：{id(conn)}（手动模式）")
            return conn
        except SQLAlchemyError as e:
            logger.error(f"数据库连接创建失败：{str(e)}")
            raise

    def _init_tables(self):
        """
        私有方法：初始化数据库表结构（不存在则创建）
        包含表：t_prediction_parameters（核心参数）、t_training_records（训练记录）
                t_geo_fault（断层信息）、t_coal_point（断层组成点）
        """
        try:
            with self._get_connection() as conn:
                # Step 1: 创建核心参数表 t_prediction_parameters，移除了gas_analysis_index_dh2 FLOAT COMMENT '瓦斯解析指数Δh2（Pa）',
                create_pred_sql = text("""
                CREATE TABLE IF NOT EXISTS t_prediction_parameters (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
                    working_face VARCHAR(50) NOT NULL COMMENT '工作面名称',
                    workface_id INT COMMENT '工作面ID（关联t_geo_fault）',
                    roadway VARCHAR(50) COMMENT '巷道名称',
                    roadway_id VARCHAR(20) COMMENT '巷道ID',
                    distance_from_entrance FLOAT COMMENT '距入口距离（m）',
                    x_coord FLOAT COMMENT 'X坐标（m）',
                    y_coord FLOAT COMMENT 'Y坐标（m）',
                    z_coord FLOAT COMMENT 'Z坐标（高程，m）',
                    coal_thickness FLOAT COMMENT '煤层厚度（m）',
                    fault_influence_strength FLOAT COMMENT '断层影响系数（系统自动计算）',
                    regional_measure_strength INT COMMENT '区域措施强度（需提前计算）',
                    tunneling_speed FLOAT COMMENT '掘进速度（m/min）',
                    roadway_length FLOAT COMMENT '巷道长度（m）',
                    initial_gas_emission_strength FLOAT COMMENT '初始瓦斯涌出强度',
                    roadway_cross_section FLOAT COMMENT '巷道断面积（m²）',
                    coal_density FLOAT COMMENT '煤密度（t/m³）',
                    original_gas_content FLOAT COMMENT '原始瓦斯含量（m³/t）',
                    residual_gas_content FLOAT COMMENT '残余瓦斯含量（m³/t）',
                    gas_emission_q FLOAT COMMENT '瓦斯涌出量Q（m³/min）',
                    drilling_cuttings_s FLOAT COMMENT '钻屑量S（kg/m）',
                    gas_emission_velocity_q FLOAT COMMENT '瓦斯涌出速度q（L/min·m）',
                    
                    gas_emission_wall FLOAT COMMENT '煤壁瓦斯涌出量（m³/min）',
                    gas_emission_fallen FLOAT COMMENT '落煤瓦斯涌出量（m³/min）',
                    total_gas_emission FLOAT COMMENT '总瓦斯涌出量（m³/min）',
                    measurement_date DATE COMMENT '测量日期（YYYY-MM-DD）',
                    depth_from_face FLOAT COMMENT '验证孔深度/测点距工作面距离（m）',
                    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    INDEX idx_working_face (working_face),
                    INDEX idx_roadway (roadway_id),
                    INDEX idx_create_time (create_time),
                    INDEX idx_measurement_date (measurement_date) COMMENT '测量日期索引',
                    INDEX idx_depth_from_face (depth_from_face) COMMENT '验证孔深度索引'
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='瓦斯预测核心参数表（简化时间特征）';
                """)
                conn.execute(create_pred_sql)
                logger.debug("表 t_prediction_parameters 初始化完成（已存在则跳过）")

                # Step 2: 创建训练记录表 t_training_records
                create_record_sql = text("""
                CREATE TABLE IF NOT EXISTS t_training_records (
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
                """)
                conn.execute(create_record_sql)
                logger.debug("表 t_training_records 初始化完成（已存在则跳过）")

                # Step 3: 创建断层信息表 t_geo_fault
                create_fault_sql = text("""
                CREATE TABLE IF NOT EXISTS t_geo_fault (
                    id INT NOT NULL AUTO_INCREMENT COMMENT '自增主键',
                    workface_id INT NOT NULL COMMENT '工作面ID（关联t_prediction_parameters）',
                    name VARCHAR(50) NOT NULL COMMENT '断层名称（如F1断层）',
                    length DECIMAL(10,2) DEFAULT NULL COMMENT '断层长度（m）',
                    azimuth DECIMAL(10,2) DEFAULT NULL COMMENT '方位角（°）',
                    inclination DECIMAL(10,2) DEFAULT NULL COMMENT '倾角（°）',
                    fault_height DECIMAL(10,2) DEFAULT NULL COMMENT '断距（m）',
                    fault_type TINYINT(4) DEFAULT NULL COMMENT '断层类型（1=正断层，2=逆断层）',
                    influence_scope DECIMAL(10,2) NOT NULL DEFAULT 0.00 COMMENT '影响范围（m）',
                    according_to VARCHAR(255) DEFAULT NULL COMMENT '断层依据（如地质勘探报告）',
                    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    PRIMARY KEY (id),
                    KEY idx_workface_id (workface_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='断层基础信息表';
                """)
                conn.execute(create_fault_sql)
                logger.debug("表 t_geo_fault 初始化完成（已存在则跳过）")

                # Step 4: 创建断层组成点表 t_coal_point（级联删除）
                create_coal_point_sql = text("""
                CREATE TABLE IF NOT EXISTS t_coal_point (
                    id INT NOT NULL AUTO_INCREMENT COMMENT '自增主键',
                    geofault_id INT NOT NULL COMMENT '关联断层ID（t_geo_fault.id）',
                    floor_coordinate_x DECIMAL(10,2) NOT NULL COMMENT 'X坐标（m）',
                    floor_coordinate_y DECIMAL(10,2) NOT NULL COMMENT 'Y坐标（m）',
                    floor_coordinate_z DECIMAL(10,2) DEFAULT 0.00 COMMENT 'Z坐标（m，默认为0）',
                    point_order INT NOT NULL COMMENT '点顺序（用于确定断层走向）',
                    workface_id INT NOT NULL COMMENT '工作面ID',
                    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    PRIMARY KEY (id),
                    KEY fk_geofault (geofault_id),
                    CONSTRAINT fk_geofault FOREIGN KEY (geofault_id) 
                        REFERENCES t_geo_fault (id) ON DELETE CASCADE,
                    CONSTRAINT uk_point_order UNIQUE KEY (geofault_id, point_order)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='断层组成点集合表';
                """)
                conn.execute(create_coal_point_sql)
                logger.debug("表 t_coal_point 初始化完成（已存在则跳过）")

                # Step 5: 动态添加配置的字段（如new_feature1）
                self._add_dynamic_columns(conn)

                # Step 6: 提交事务
                conn.commit()
                logger.debug("所有表结构初始化完成")
        except Exception as e:
            logger.error(f"表结构初始化失败：{str(e)}", exc_info=True)
            raise

    def _add_dynamic_columns(self, conn):
        """
        私有方法：为t_prediction_parameters添加动态字段（从config.ini读取）

        :param conn: sqlalchemy.engine.Connection，数据库连接对象
        """
        try:
            # Step 1: 检查t_prediction_parameters是否存在
            table_exists = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = DATABASE() AND table_name = 't_prediction_parameters'
            """)).scalar()
            if not table_exists:
                logger.warning("表 t_prediction_parameters 不存在，跳过动态字段添加")
                return

            # Step 2: 获取现有字段列表
            existing_cols = conn.execute(text("""
                SELECT COLUMN_NAME FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 't_prediction_parameters'
            """)).fetchall()
            existing_col_names = {row[0] for row in existing_cols}

            # Step 3: 添加缺失的动态字段
            for col_name, col_type in self.dynamic_columns.items():
                if col_name not in existing_col_names:
                    add_col_sql = text(f"ALTER TABLE t_prediction_parameters ADD COLUMN {col_name} {col_type} COMMENT '动态字段'")
                    conn.execute(add_col_sql)
                    logger.debug(f"动态添加字段：{col_name}（类型：{col_type}）")
                else:
                    logger.debug(f"动态字段 {col_name} 已存在，跳过")
        except Exception as e:
            logger.error(f"添加动态字段失败：{str(e)}，不影响主流程")

    @retry(
        stop=stop_after_attempt(3),  # 重试3次
        wait=wait_exponential(multiplier=1, min=4, max=10)  # 指数退避等待（4s→8s→10s）
    )
    def save_training_data(self, df, column_mapping=None, conn=None, trans=None, custom_create_time=None):
        """
        批量插入训练数据到t_prediction_parameters（支持外部事务，确保原子性）

        :param df: pandas.DataFrame，待插入的数据（需包含核心字段）
        :param column_mapping: dict，字段映射（key=df列名，value=数据库列名），默认None
        :param conn: sqlalchemy.engine.Connection，外部连接（用于事务控制），默认None
        :param trans: sqlalchemy.engine.Transaction，外部事务（与conn配套），默认None
        :param custom_create_time: str，自定义创建时间（如"2025-10-15 16:30:00.123"），默认None
        :return: int，成功插入的记录数
        """
        # Step 1: 校验输入数据
        if df is None or df.empty:
            logger.warning("无数据需保存到数据库（df为空）")
            return 0
        mapped_df = df.copy()

        # Step 2: 应用字段映射（解决df列名与数据库列名不匹配）
        if column_mapping:
            valid_cols = [col for col in mapped_df.columns if col in column_mapping]
            mapped_df = mapped_df[valid_cols].rename(columns=column_mapping)
            logger.debug(f"应用字段映射后保留列：{mapped_df.columns.tolist()}")

        # Step 3: 确定数据库有效列（核心字段+动态字段）移除了'gas_analysis_index_dh2',
        db_core_cols = [
            'working_face', 'workface_id', 'roadway', 'roadway_id',
            'distance_from_entrance', 'x_coord', 'y_coord', 'z_coord',
            'coal_thickness', 'regional_measure_strength', 'tunneling_speed',
            'roadway_length', 'initial_gas_emission_strength', 'roadway_cross_section',
            'coal_density', 'original_gas_content', 'residual_gas_content',
            'gas_emission_q', 'drilling_cuttings_s', 'gas_emission_velocity_q',
            'gas_emission_wall', 'gas_emission_fallen',
            'total_gas_emission',
            # 简化版时间相关字段
            'measurement_date', 'depth_from_face'
        ]
        db_valid_cols = db_core_cols + list(self.dynamic_columns.keys())
        # 过滤df中存在的有效列
        insert_cols = [col for col in mapped_df.columns if col in db_valid_cols]
        if not insert_cols:
            logger.warning("无匹配的列可插入数据库（检查字段映射是否正确）")
            return 0

        # Step 4: 处理自定义创建时间（用于事务回滚时定位数据）
        if custom_create_time:
            mapped_df['create_time'] = custom_create_time
            insert_cols.append('create_time')
            logger.debug(f"添加自定义create_time：{custom_create_time}")

        # Step 5: 生成插入SQL和记录（参数化查询，避免SQL注入）
        insert_sql = text(f"""
            INSERT INTO t_prediction_parameters ({', '.join(insert_cols)})
            VALUES ({', '.join([':' + col for col in insert_cols])})
        """)
        records = mapped_df[insert_cols].to_dict(orient='records')
        if not records:
            logger.warning("数据转换为dict失败（无有效记录）")
            return 0

        # Step 6: 执行插入（优先使用外部事务，无则用内部事务）
        try:
            if conn and trans:
                # 外部事务模式（如train方法中调用，由外部控制提交/回滚）
                conn.execute(insert_sql, records)
                saved_count = len(records)
                logger.debug(f"外部事务中插入数据：{saved_count}条（未提交）")
                return saved_count
            else:
                # 内部事务模式（单独调用时，自动提交/回滚）
                with self._get_connection() as inner_conn:
                    inner_trans = inner_conn.begin()
                    try:
                        inner_conn.execute(insert_sql, records)
                        inner_trans.commit()
                        saved_count = len(records)
                        logger.info(f"内部事务中保存数据：{saved_count}条（已提交）")
                        return saved_count
                    except Exception as e:
                        inner_trans.rollback()
                        logger.error(f"内部事务插入失败：{str(e)}，已回滚")
                        return 0
        except Exception as e:
            logger.error(f"保存训练数据失败：{str(e)}", exc_info=True)
            return 0

    def insert_training_record(self, record):
        """
        插入训练记录到t_training_records表

        :param record: dict，训练记录，需包含：
            sample_count: int，本次训练样本数
            total_samples: int，累计样本数
            train_mode: str，训练模式（全量/增量）
            status: str，训练状态（success/warning/error）
            message: str，训练信息
            duration: float，训练耗时（秒）
            train_time: datetime，训练时间（可选，默认CURRENT_TIMESTAMP）
        :return: int/None，成功则返回记录ID，失败则返回None
        """
        if not record:
            logger.warning("训练记录为空，不插入数据库")
            return None

        # 构造插入数据（处理可选字段）
        insert_data = {
            "sample_count": record.get("sample_count", 0),
            "total_samples": record.get("total_samples", 0),
            "train_mode": record.get("train_mode", "未知"),
            "status": record.get("status", "unknown"),
            "message": record.get("message", ""),
            "duration": record.get("duration", 0.0),
            "train_time": record.get("train_time") or None
        }

        try:
            with self._get_connection() as conn:
                insert_sql = text("""
                    INSERT INTO t_training_records 
                    (sample_count, total_samples, train_mode, status, message, duration, train_time)
                    VALUES (:sample_count, :total_samples, :train_mode, :status, :message, :duration, :train_time)
                """)
                result = conn.execute(insert_sql, insert_data)
                conn.commit()
                record_id = result.lastrowid
                logger.debug(f"插入训练记录成功，ID：{record_id}")
                return record_id
        except Exception as e:
            logger.error(f"插入训练记录失败：{str(e)}", exc_info=True)
            return None

    def get_faults_by_workface(self, workface_id):
        """
        按工作面ID查询所有断层信息（t_geo_fault表）

        :param workface_id: int，工作面ID
        :return: list[dict]，断层列表，每个元素包含断层ID、名称、长度、倾角等
        """
        try:
            with self._get_connection() as conn:
                query_sql = text("""
                    SELECT id, name, length, azimuth, inclination, 
                           fault_height, fault_type, influence_scope 
                    FROM t_geo_fault 
                    WHERE workface_id = :workface_id
                """)
                result = conn.execute(query_sql, {"workface_id": workface_id})
                faults = [dict(row._mapping) for row in result.fetchall()]
                logger.debug(f"查询工作面{workface_id}的断层：{len(faults)}条")
                return faults
        except Exception as e:
            logger.error(f"查询工作面{workface_id}断层失败：{str(e)}", exc_info=True)
            return []

    def get_fault_points(self, geofault_id):
        """
        按断层ID查询所有组成点（t_coal_point表），按point_order排序

        :param geofault_id: int，断层ID（t_geo_fault.id）
        :return: list[dict]，组成点列表，每个元素包含X/Y/Z坐标
        """
        try:
            with self._get_connection() as conn:
                query_sql = text("""
                    SELECT floor_coordinate_x, floor_coordinate_y, floor_coordinate_z 
                    FROM t_coal_point 
                    WHERE geofault_id = :geofault_id
                    ORDER BY point_order ASC
                """)
                result = conn.execute(query_sql, {"geofault_id": geofault_id})
                points = [dict(row._mapping) for row in result.fetchall()]
                logger.debug(f"查询断层{geofault_id}的组成点：{len(points)}条")
                return points
        except Exception as e:
            logger.error(f"查询断层{geofault_id}组成点失败：{str(e)}", exc_info=True)
            return []