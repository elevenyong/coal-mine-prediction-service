"""
煤矿瓦斯风险预测系统 - 模型评估模块
包含：模型性能评估、性能监控、自动回滚
依赖：scikit-learn、numpy
"""
import pandas as pd
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
from sqlalchemy import create_engine, text

from config_utils import ConfigUtils


class ModelEvaluator(ConfigUtils):
    """模型评估器"""

    def __init__(self, config_path="config.ini"):
        super().__init__(config_path)
        self.eval_size = self.config.getint("ModelEval", "eval_size", fallback=200)
        # 评估集划分策略：
        # - latest：沿用旧逻辑（按distance_from_entrance排序取最近N条）
        # - time_holdout：按measurement_date留出最后N天作为评估集（更可信）
        self.eval_split_mode = self.config.get("ModelEval", "eval_split_mode", fallback="latest").strip().lower()
        self.holdout_days = self.config.getint("ModelEval", "holdout_days", fallback=3)
        self.min_holdout_samples = self.config.getint("ModelEval", "min_holdout_samples", fallback=20)
        # ====== 新增：评估集“去泄漏”控制（全部有fallback，不要求你改config）======
        # group_holdout_enabled：time_holdout时，评估集尽量选择“训练期未出现过的 group”
        self.group_holdout_enabled = self.config.getboolean("ModelEval", "group_holdout_enabled", fallback=True)
        # group_key：优先用 spatiotemporal_group（已把 coord_hash + distance_to_face_bucket 融合进来），其次 coord_hash
        self.eval_group_key = self.config.get("ModelEval", "eval_group_key", fallback="spatiotemporal_group").strip()
        # 当日期跨度太短（例如只有1~2天），time_holdout会失真；short_span_backoff=True 时会自动降级为 latest + 去重
        self.short_span_backoff = self.config.getboolean("ModelEval", "short_span_backoff", fallback=True)
        # 最新策略下：按 group 去重后再截取 eval_size（避免大量重复点导致评估偏置）
        self.eval_dedupe_by_group = self.config.getboolean("ModelEval", "eval_dedupe_by_group", fallback=True)
        self.perf_drop_ratio = self.config.getfloat("ModelEval", "perf_drop_ratio", fallback=0.10)
        self.perf_window = self.config.getint("ModelEval", "perf_window", fallback=2)
        self.small_data_threshold = self.config.getint("ModelEval", "small_data_threshold", fallback=10)
        self.medium_data_threshold = self.config.getint("ModelEval", "medium_data_threshold", fallback=20)
        agg_weights_str = self.config.get("ModelEval", "agg_weights", fallback="1,1,1,1")
        try:
            self.agg_weights = [float(x) for x in agg_weights_str.split(",")]
        except Exception:
            self.agg_weights = [1.0, 1.0, 1.0, 1.0]
            logger.warning(f"聚合权重解析失败（配置：{agg_weights_str}），使用默认值[1.0,1.0,1.0,1.0]")

    def evaluate_model(self, models, preprocessor, training_features, target_features,
                       fitted_feature_order, db_utils, eval_size=None, eval_df=None,
                       data_preprocessor=None):
        def _choose_group_key(df: pd.DataFrame) -> Optional[str]:
            """
            评估分组键优先级：
              1) self.eval_group_key（默认 spatiotemporal_group）
              2) spatiotemporal_group
              3) coord_hash
            """
            try:
                if self.eval_group_key and self.eval_group_key in df.columns:
                    return self.eval_group_key
            except Exception:
                pass
            for k in ["spatiotemporal_group", "coord_hash"]:
                if k in df.columns:
                    return k
            return None

        def _dedupe_eval_by_group(df: pd.DataFrame, group_key: Optional[str]) -> pd.DataFrame:
            """
            评估集去重：同一个 group 只保留“最近一条”（按 measurement_date、id、distance_from_entrance 的综合排序）
            目的：避免评估集被单个坐标点/孔深桶的密集记录刷屏，导致指标不稳定/偏置。
            """
            if df is None or df.empty or not group_key or group_key not in df.columns:
                return df
            try:
                sort_cols = []
                if "measurement_date" in df.columns:
                    sort_cols.append("measurement_date")
                if "id" in df.columns:
                    sort_cols.append("id")
                if "distance_from_entrance" in df.columns:
                    sort_cols.append("distance_from_entrance")
                if sort_cols:
                    df = df.sort_values(sort_cols, ascending=False, kind="mergesort")
                # 每组取第一条（最近）
                return df.drop_duplicates(subset=[group_key], keep="first")
            except Exception as _e:
                logger.warning(f"评估集按group去重失败（已忽略）：{repr(_e)}", exc_info=True)
                return df
        """
        公开方法：模型评估（计算RMSE/MAE/R²，支持外部传入评估数据）

        :param models: 模型字典
        :param preprocessor: 特征预处理器
        :param training_features: 训练特征列表
        :param target_features: 目标特征列表
        :param fitted_feature_order: 训练时的特征顺序
        :param db_utils: 数据库工具实例
        :param eval_size: 评估样本数
        :param eval_df: 外部评估数据
        :return: dict，评估结果
        """
        self._print_header("模型性能评估")
        try:
            if not models:
                raise ValueError("模型字典不能为空")
            if preprocessor is None:
                raise ValueError("特征预处理器不能为None")
            if not training_features:
                raise ValueError("训练特征列表不能为空")
            if not target_features:
                raise ValueError("目标特征列表不能为空")
            if eval_df is None:
                eval_size = eval_size or self.eval_size
                if eval_size <= 0:
                    raise ValueError(f"评估样本数必须为正数，当前值：{eval_size}")
                db_conf = db_utils.db_config
                db_url = f"mysql+pymysql://{db_conf['user']}:{db_conf['password']}@" \
                         f"{db_conf['host']}:{db_conf['port']}/{db_conf['db']}?charset={db_conf['charset']}"
                engine = create_engine(db_url)
                # 修复：不要再用 p.*（p表里可能保留了与f表同名的增强特征列，导致重复列名）
                # 改为：按“训练所需 + 目标 + 最少索引”动态显式选择 p 列，并显式选择 f 的增强特征列。
                # 这样可以从根上消灭 duplicate labels，而不是依赖 pandas 事后去重。
                p_index_cols = [
                    "id", "working_face", "workface_id", "work_stage",
                    "roadway", "roadway_id", "measurement_date",
                    "borehole_id", "x_coord", "y_coord", "z_coord",
                    "distance_to_face", "distance_from_entrance"
                ]

                # 统一列清单生成器：以“训练所需 + 目标 + 最少索引”为输入，按 DB schema 生成 p/f 两张表的最终列清单
                needed_cols = []
                for c in (p_index_cols + list(training_features) + list(target_features)):
                    if c and c not in needed_cols:
                        needed_cols.append(c)

                # 防注入：只允许字母/数字/下划线的列名（双保险）
                def _safe_col(col: str) -> bool:
                    return isinstance(col, str) and col.replace("_", "").isalnum()

                # 由 DBUtils 统一生成列清单（根治重复列名 + Unknown column）
                try:
                    if db_utils is not None and hasattr(db_utils, "build_join_column_lists"):
                        p_cols, f_cols = db_utils.build_join_column_lists(needed_cols=needed_cols,
                                                                          include_targets=False)
                    else:
                        # 极端兜底：无 DBUtils/旧版本时，退回到“最小默认列”策略
                        p_cols, f_cols = needed_cols, [
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
                except Exception as _e:
                    logger.warning(f"评估取数：统一列清单生成失败（已回退到最小列集合）：{repr(_e)}", exc_info=True)
                    p_cols, f_cols = needed_cols, []

                p_select_cols = [f"p.{c}" for c in p_cols if _safe_col(c)]
                f_select_cols = [f"f.{c} AS {c}" for c in f_cols if _safe_col(c)]
                select_sql = ",\n".join(p_select_cols + f_select_cols)
                # 为支持 time_holdout，需要先取一批候选数据再按日期筛选
                candidate_limit = max(int(eval_size * 3), int(eval_size))
                # 关键修复：部分环境下 LIMIT 绑定参数可能触发 KeyError('limit') / driver 不兼容
                # 这里将 limit 作为安全的整数字面量拼接（已强制 int），避免绑定失败
                limit_literal = int(candidate_limit)
                # ===== 最后一刀：评估SQL兜底，禁止引用 p.advance_rate_mining（主表无该列）=====
                try:
                    if isinstance(select_sql, str) and "p.advance_rate_mining" in select_sql:
                        # 删除这个字段（兼容带/不带逗号、带空格等情况）
                        select_sql = select_sql.replace("p.advance_rate_mining,", "")
                        select_sql = select_sql.replace("p.advance_rate_mining", "")
                        select_sql = select_sql.replace(",\n\n", ",\n")  # 简单清理多余空行
                        logger.warning("评估SQL兜底：已从 select_sql 移除 p.advance_rate_mining（避免 Unknown column）")
                except Exception as _e:
                    logger.warning(f"评估SQL兜底修复失败（已忽略）：{repr(_e)}", exc_info=True)
                # ===================================================================

                query = text(f"""
                    -- 分表方案评估取数（根治重复列名）：
                    -- 1）核心样本/目标值：来自 t_prediction_parameters（仅取训练所需+目标+最少索引）
                    -- 2）增强/趋势/统计特征：来自 t_feature_cache（通过pred_id关联）
                    SELECT
                        {select_sql}
                    FROM t_prediction_parameters p
                    LEFT JOIN t_feature_cache f
                      ON p.id = f.pred_id
                    ORDER BY p.measurement_date DESC, p.id DESC
                    LIMIT {limit_literal}
                """)
                with engine.connect() as conn:
                    eval_df = pd.read_sql(query, conn)
                eval_df.columns = eval_df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
                # 关键修复：若训练特征仍包含 advance_rate_mining（历史配置/已fit顺序需要），
                # 则用 f.advance_rate（已查询为 advance_rate）回填生成 advance_rate_mining，避免后续构建X缺列。
                try:
                    if ("advance_rate_mining" in list(training_features)) and (
                            "advance_rate_mining" not in eval_df.columns):
                        if "advance_rate" in eval_df.columns:
                            eval_df["advance_rate_mining"] = eval_df["advance_rate"]
                        else:
                            eval_df["advance_rate_mining"] = 0.0
                except Exception:
                    pass
                # ====== 新增：按 measurement_date 留出最后N天作为评估集（更可信）======
                if self.eval_split_mode == "time_holdout":
                    if "measurement_date" not in eval_df.columns:
                        logger.warning("eval_split_mode=time_holdout，但评估数据缺少measurement_date，回退为latest策略")
                    else:
                        dt = pd.to_datetime(eval_df["measurement_date"], errors="coerce")
                        valid_mask = dt.notna()
                        if valid_mask.sum() == 0:
                            logger.warning("time_holdout：measurement_date解析失败（全为NaT），回退为latest策略")
                        else:
                            min_dt = dt[valid_mask].min()
                        max_dt = dt[valid_mask].max()
                        span_days = int((max_dt - min_dt).days) if (min_dt is not None and max_dt is not None) else 0
                        uniq_dates = int(dt[valid_mask].dt.date.nunique())
                        # 若数据跨度过短，time_holdout 形同虚设：自动降级到 latest + group 去重（可配置关闭）
                        if self.short_span_backoff and (span_days < max(self.holdout_days, 1) or uniq_dates <= 2):
                            logger.warning(
                                f"time_holdout降级：日期跨度过短 span_days={span_days}, uniq_dates={uniq_dates}，"
                                f"holdout_days={self.holdout_days}。将回退为 latest + (可选)group去重"
                            )
                        else:
                            cutoff = max_dt - pd.Timedelta(days=max(self.holdout_days, 1))
                            holdout_df = eval_df[valid_mask & (dt >= cutoff)].copy()
                            # ====== P0：time_holdout + group_holdout（避免训练期/评估期 group 重叠导致指标虚高/虚低）======
                            group_key = _choose_group_key(eval_df)
                            if self.group_holdout_enabled and group_key and group_key in eval_df.columns:
                                try:
                                    train_period_df = eval_df[valid_mask & (dt < cutoff)].copy()
                                    train_groups = set(
                                        train_period_df[group_key].dropna().astype(str).unique().tolist()
                                    )
                                    before_n = len(holdout_df)
                                    holdout_df2 = holdout_df[
                                        ~holdout_df[group_key].astype(str).isin(train_groups)
                                    ].copy()
                                    after_n = len(holdout_df2)

                                    # 若严格 group_holdout 导致样本过少，则回退到“纯时间 holdout”（但仍可做组内去重）
                                    if after_n >= self.min_holdout_samples:
                                        holdout_df = holdout_df2
                                        logger.info(
                                            f"time_holdout+group_holdout启用：group_key={group_key}，"
                                            f"过滤训练期重复group：{before_n} → {after_n}（cutoff={cutoff.date()}）"
                                        )
                                    else:
                                        logger.warning(
                                            f"time_holdout+group_holdout样本不足：{after_n} < min_holdout_samples={self.min_holdout_samples}，"
                                            f"回退为纯time_holdout（cutoff={cutoff.date()}）"
                                        )
                                except Exception as _e:
                                    logger.warning(f"group_holdout处理失败（已忽略，继续纯time_holdout）：{repr(_e)}",
                                                   exc_info=True)

                            # （可选）评估集内部按group去重：同一group只留最近一条
                            group_key2 = _choose_group_key(holdout_df)
                            if self.eval_dedupe_by_group:
                                holdout_df = _dedupe_eval_by_group(holdout_df, group_key2)

                            if len(holdout_df) < self.min_holdout_samples:
                                logger.warning(
                                    f"time_holdout：最后{self.holdout_days}天(含去重/过滤)样本数={len(holdout_df)} "
                                    f"< 最小阈值{self.min_holdout_samples}，回退为latest策略"
                                )
                            else:
                                # 保持“最近在前”的工程语义：仍按distance_from_entrance排序后取头部
                                if "distance_from_entrance" in holdout_df.columns:
                                    holdout_df = holdout_df.sort_values("distance_from_entrance", ascending=False)
                                eval_df = holdout_df.head(eval_size).copy()
                                logger.info(
                                    f"time_holdout评估集启用：holdout_days={self.holdout_days}，"
                                    f"实际评估样本数={len(eval_df)}，max_date={max_dt.date()}, cutoff={cutoff.date()}，"
                                    f"span_days={span_days}, uniq_dates={uniq_dates}"
                                )

                # 如果未启用time_holdout或回退，则沿用latest：按distance_from_entrance取前eval_size条
                if len(eval_df) > eval_size:
                    if "distance_from_entrance" in eval_df.columns:
                        eval_df = eval_df.sort_values("distance_from_entrance", ascending=False).head(eval_size).copy()
                    else:
                        eval_df = eval_df.head(eval_size).copy()
                # latest / 回退路径：可选按 group 去重（防止评估集被同一点/同桶刷屏）
                try:
                    if self.eval_dedupe_by_group and eval_df is not None and not eval_df.empty:
                        gk = _choose_group_key(eval_df)
                        before_n = len(eval_df)
                        eval_df = _dedupe_eval_by_group(eval_df, gk)
                        after_n = len(eval_df)
                        if gk and after_n != before_n:
                            logger.info(f"评估集latest去重：group_key={gk}，{before_n} → {after_n}")
                        # 去重后再截断到 eval_size
                        if len(eval_df) > eval_size:
                            if "distance_from_entrance" in eval_df.columns:
                                eval_df = eval_df.sort_values("distance_from_entrance", ascending=False).head(eval_size).copy()
                            else:
                                eval_df = eval_df.head(eval_size).copy()
                except Exception as _e:
                    logger.warning(f"评估集latest去重失败（已忽略）：{repr(_e)}", exc_info=True)
            if eval_df.empty:
                msg = "未获取到评估样本，跳过评估"
                self._print_result(msg)
                return {"status": "warning", "message": msg}
            self._print_step(f"评估样本数：{len(eval_df)}")
            # 检查目标列
            missing_targets = [t for t in target_features if t not in eval_df.columns]
            if missing_targets:
                msg = f"评估数据缺少目标列：{missing_targets}，跳过评估"
                logger.warning(msg)
                self._print_result(msg)
                return {"status": "warning", "message": msg}

            # 评估阶段说明：
            # - 分表方案下，days_*、bucket、trend、historical_mean 等增强特征优先来自 t_feature_cache
            # - 评估阶段再次调用 DataPreprocessor 生成特征，容易引入pandas布尔歧义错误（Series truth value ambiguous）
            # - 因此评估阶段默认不再重复跑 preprocess_data，仅做“缺失特征补齐 + 顺序对齐”
            # 评估输入特征必须与 preprocessor.fit 时一致，否则会出现：
            # X has N features, but ColumnTransformer is expecting M features as input
            # 先处理：eval_df 列名重复会导致 reindex 直接失败（cannot reindex on an axis with duplicate labels）
            try:
                dup_mask = eval_df.columns.duplicated(keep=False)
                if dup_mask.any():
                    dup_cols = eval_df.columns[dup_mask].tolist()
                    # 只打印去重后的集合，避免日志过长
                    dup_unique = sorted(set(dup_cols))
                    logger.error(f"评估数据存在重复列名（将保留首个并丢弃后续重复列）：{dup_unique}")
                    # 去重：保留第一次出现的列
                    eval_df = eval_df.loc[:, ~eval_df.columns.duplicated(keep="first")]
            except Exception as _e:
                logger.warning(f"评估数据重复列检测/去重失败：{repr(_e)}", exc_info=True)
            expected_cols = None
            if hasattr(preprocessor, "feature_names_in_") and preprocessor.feature_names_in_ is not None:
                expected_cols = list(preprocessor.feature_names_in_)
            else:
                # 兼容：旧版本预处理器没有 feature_names_in_，退回使用训练时记录的顺序
                expected_cols = list(fitted_feature_order) if fitted_feature_order else list(training_features)
            # ============ 新增：增强特征缺失率/填零率诊断 ============
            # 目的：
            # - 明确 t_feature_cache 是否为每条样本生成了增强特征
            # - 明确哪些增强特征大量缺失/被填0（会显著拉低评估指标，导致R²为负）
            try:
                n_rows = len(eval_df)
                if n_rows > 0:
                    # 这些字段是你日志中出现重复的“增强特征”，也最容易缺失
                    enhanced_cols = [
                        "coord_hash",
                        "spatiotemporal_group",
                        "distance_to_face_bucket",
                        "days_since_start",
                        "days_in_workface",
                        "distance_time_interaction",
                        "drilling_cuttings_s_historical_mean",
                        "gas_emission_velocity_q_historical_mean",
                        "drilling_cuttings_s_trend",
                        "gas_emission_velocity_q_trend",
                        "advance_rate",
                    ]
                    def _fmt(p):
                        return f"{p * 100:.1f}%"
                    diag_lines = []
                    for c in enhanced_cols:
                        if c not in eval_df.columns:
                            diag_lines.append(f"  - {c}: 缺列（100.0%）")
                            continue
                        s = eval_df[c]
                        # 缺失率：NaN/None
                        miss_ratio = float(s.isna().mean())
                        # 填零率：仅对数值列统计（object列如hash不统计0）
                        if pd.api.types.is_numeric_dtype(s):
                            zero_ratio = float((s.fillna(0) == 0).mean())
                            diag_lines.append(f"  - {c}: 缺失={_fmt(miss_ratio)}, 为0={_fmt(zero_ratio)}")
                        else:
                            diag_lines.append(f"  - {c}: 缺失={_fmt(miss_ratio)}")
                    logger.info("=== 评估输入增强特征缺失诊断（来自t_feature_cache） ===")
                    for line in diag_lines:
                        logger.info(line)
                    # 再对“预处理器期望输入特征”做整体缺失概览（缺列/缺失/为0）
                    miss_cols = [c for c in expected_cols if c not in eval_df.columns]
                    if miss_cols:
                        logger.warning(f"评估数据缺少预处理器期望特征列 {len(miss_cols)} 个（将被补0）：{miss_cols}")
                    # 对存在的数值特征统计“缺失率均值/填零率均值”
                    exist_cols = [c for c in expected_cols if c in eval_df.columns]
                    if exist_cols:
                        numeric_exist = [c for c in exist_cols if pd.api.types.is_numeric_dtype(eval_df[c])]
                        if numeric_exist:
                            miss_mean = float(eval_df[numeric_exist].isna().mean().mean())
                            zero_mean = float((eval_df[numeric_exist].fillna(0) == 0).mean().mean())
                            logger.info(
                                f"=== 评估输入总体概览（仅数值列，n={len(numeric_exist)}） === "
                                f"平均缺失率={_fmt(miss_mean)}，平均为0率={_fmt(zero_mean)}"
                            )
            except Exception as _e:
                logger.warning(f"评估输入缺失诊断失败：{repr(_e)}", exc_info=True)
            # =================== P0：评估输入有效性判定（不可信则返回 evaluation_invalid） ===================
            # 触发场景：t_feature_cache 未真正生成增强特征（大量字段“为0=100%/缺列”），
            # 此时算出来的 RMSE/R² 往往是“假的好/假的坏”，会误导基线与回滚策略。
            try:
                # 复用上面 enhanced_cols 列表（如果作用域里不存在就本地再声明一次）
                if "enhanced_cols" not in locals():
                    enhanced_cols = [
                        "coord_hash",
                        "spatiotemporal_group",
                        "distance_to_face_bucket",
                        "days_since_start",
                        "days_in_workface",
                        "distance_time_interaction",
                        "drilling_cuttings_s_historical_mean",
                        "gas_emission_velocity_q_historical_mean",
                        "drilling_cuttings_s_trend",
                        "gas_emission_velocity_q_trend",
                        "advance_rate",
                    ]

                invalid_reasons = []
                enhanced_stats = {}

                # 统计增强特征：缺列/缺失率/为0率
                for c in enhanced_cols:
                    if c not in eval_df.columns:
                        enhanced_stats[c] = {"missing_col": True, "missing_ratio": 1.0, "zero_ratio": None}
                        continue
                    s = eval_df[c]
                    miss_ratio = float(s.isna().mean())
                    zero_ratio = None
                    if pd.api.types.is_numeric_dtype(s):
                        zero_ratio = float((s.fillna(0) == 0).mean())
                    enhanced_stats[c] = {"missing_col": False, "missing_ratio": miss_ratio, "zero_ratio": zero_ratio}

                # 关键增强数值列（这些基本都应“不是全0”）
                critical_numeric = [
                    "days_since_start",
                    "days_in_workface",
                    "distance_time_interaction",
                    "drilling_cuttings_s_historical_mean",
                    "gas_emission_velocity_q_historical_mean",
                    "drilling_cuttings_s_trend",
                    "gas_emission_velocity_q_trend",
                    "advance_rate",
                ]
                critical_present = [c for c in critical_numeric if
                                    c in enhanced_stats and not enhanced_stats[c]["missing_col"]]
                critical_bad = []
                for c in critical_present:
                    zr = enhanced_stats[c].get("zero_ratio", None)
                    mr = enhanced_stats[c].get("missing_ratio", 0.0)
                    # 判定“坏”：为0率>=95% 且缺失率不高（说明是被填0/没算出来，而非真实缺失）
                    if zr is not None and float(zr) >= 0.95 and float(mr) <= 0.50:
                        critical_bad.append(c)

                # 规则：关键增强特征中 >=3 个“近乎全0”，则评估输入不可信
                if len(critical_bad) >= 3:
                    invalid_reasons.append(
                        f"关键增强特征近乎全0（>=95%）的列数={len(critical_bad)}：{critical_bad}"
                    )

                # 规则：预处理器期望列存在大量缺列（会被补0），评估不可信
                miss_cols = [c for c in expected_cols if c not in eval_df.columns]
                if len(expected_cols) > 0 and (len(miss_cols) / float(len(expected_cols))) >= 0.30:
                    invalid_reasons.append(
                        f"评估输入缺列比例过高：missing={len(miss_cols)}/{len(expected_cols)}（>=30%）"
                    )

                if invalid_reasons:
                    msg = "评估输入不可信（增强特征未有效生成/大量被补0），本次评估标记为 evaluation_invalid"
                    logger.warning(msg + "；原因：" + "；".join(invalid_reasons))

                    # 明确整改建议：不掩盖问题
                    suggest = [
                        "检查 t_feature_cache 是否在训练/入库时为每条 pred_id 生成了增强特征（trend/historical_mean/advance_rate 等）。",
                        "训练阶段 bridge/getter 请尽量传 measurement_date，避免取到“未来增强特征”。",
                        "若 feature_cache 逻辑尚未跑通：应先修复特征生成闭环，再看 RMSE/R²，否则指标无意义。",
                    ]

                    self._print_result(msg)
                    return {
                        "status": "warning",
                        "code": "evaluation_invalid",
                        "message": msg,
                        "details": {
                            "reasons": invalid_reasons,
                            "suggestions": suggest,
                            "enhanced_feature_stats": enhanced_stats,
                            "missing_expected_cols_count": len(miss_cols),
                            "missing_expected_cols": miss_cols[:50],  # 避免太长
                        },
                        "avg_rmse": None,
                        "per_target": None,
                        "sample_count": int(len(eval_df)) if eval_df is not None else 0,
                    }
            except Exception as _e:
                logger.warning(f"评估输入有效性判定失败（忽略，不影响后续正常评估流程）：{repr(_e)}", exc_info=True)
            # ===============================================================================================

            # 诊断：训练特征列表与预处理器期望不一致时，明确提示（这是真问题，不要掩盖）
            if training_features is not None and len(training_features) != len(expected_cols):
                logger.warning(
                    f"评估输入特征数量异常：training_features={len(training_features)}，"
                    f"preprocessor期望={len(expected_cols)}。将以preprocessor期望列为准进行对齐。"
                )
            # 强制对齐：缺列补0，多列丢弃，顺序严格按 expected_cols
            X_eval = eval_df.reindex(columns=expected_cols, fill_value=0.0)
            y_true = eval_df[target_features].values
            # 关键修复：以 preprocessor 在 fit 时记录的 feature_names_in_ 作为“最终顺序”
            # sklearn 会在 transform 阶段校验输入列名与顺序必须与 fit 一致
            expected_order = None
            if hasattr(preprocessor, "feature_names_in_") and preprocessor.feature_names_in_ is not None:
                expected_order = list(preprocessor.feature_names_in_)
            else:
                # 兼容旧版本：退回到 fitted_feature_order
                expected_order = list(fitted_feature_order) if fitted_feature_order else list(X_eval.columns)
            # 若 expected_order 中存在缺列，补0（避免直接失败）
            missing_for_expected = [c for c in expected_order if c not in X_eval.columns]
            if missing_for_expected:
                logger.warning(
                    f"评估数据缺少{len(missing_for_expected)}个预处理器期望特征，自动补0：{missing_for_expected}")
                for c in missing_for_expected:
                    X_eval[c] = 0.0
            # 重排为 expected_order（彻底解决“特征名顺序不一致”）
            if list(X_eval.columns) != expected_order:
                logger.warning("评估特征顺序与预处理器fit阶段不一致，按feature_names_in_重新排序")
                X_eval = X_eval[expected_order]
            # transform 前清洗 inf/NaN，避免 StandardScaler/统计计算出现 invalid value warning
            try:
                import numpy as np
                X_eval = X_eval.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            except Exception:
                pass

            # 特征预处理+模型预测
            # 说明：sklearn 对 DataFrame 输入会强制校验“特征名+顺序”必须与fit一致；
            # 即使我们已按 feature_names_in_ 重排，仍可能因列对象差异/重复列等触发校验。
            # 评估阶段使用 numpy 输入可跳过该校验（前提：已完成顺序对齐）。
            try:
                # 此时 X_eval 的列集合/顺序已严格对齐，直接DataFrame输入即可
                X_eval_proc = preprocessor.transform(X_eval)
            except Exception as e:
                logger.error(f"评估特征预处理失败：{repr(e)}", exc_info=True)
                raise
            if not models:
                msg = "模型未训练，无法评估"
                logger.error(msg)
                return {"status": "error", "message": msg}
            # 检测算法类型
            algorithm_type = self._detect_algorithm_type(models)
            logger.debug(f"评估使用算法类型: {algorithm_type}")
            y_pred = []
            for target in target_features:
                if target not in models:
                    raise ValueError(f"未找到目标 {target} 的模型")
                model = models[target]
                # 根据算法类型进行预测
                if algorithm_type == "lightgbm":
                    pred = model.predict(X_eval_proc)
                elif algorithm_type == "xgboost":
                    # 将数据转换为XGBoost需要的格式
                    import xgboost as xgb
                    if hasattr(X_eval_proc, 'toarray'):
                        # 如果是稀疏矩阵，转换为稠密矩阵
                        X_eval_dense = X_eval_proc.toarray()
                    else:
                        X_eval_dense = X_eval_proc
                    dmatrix = xgb.DMatrix(X_eval_dense)
                    pred = model.predict(dmatrix)
                else:
                    raise ValueError(f"不支持的算法类型: {algorithm_type}")
                y_pred.append(pred.reshape(-1, 1))
            y_pred = np.hstack(y_pred)
            # 计算评估指标
            per_target_metrics = {}
            weighted_rmse_sum = 0.0
            for i, target in enumerate(target_features):
                weight = self.agg_weights[i] if i < len(self.agg_weights) else 1.0
                rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                per_target_metrics[target] = {
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "r2": round(r2, 4),
                    "weight": weight
                }
                weighted_rmse_sum += rmse * weight

            total_weight = sum(self.agg_weights[:len(target_features)]) or 1.0
            avg_rmse = round(weighted_rmse_sum / total_weight, 4)

            self._print_step("评估指标详情：")
            for target, metrics in per_target_metrics.items():
                self._print_step(f"  {target}：RMSE={metrics['rmse']}，MAE={metrics['mae']}，R²={metrics['r2']}")
            self._print_step(f"加权平均RMSE：{avg_rmse}")

            return {
                "status": "success",
                "avg_rmse": avg_rmse,
                "per_target": per_target_metrics,
                "sample_count": len(eval_df)
            }
        except Exception as e:
            logger.error(f"评估失败：{str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _performance_trigger_check(self, eval_history, baseline_rmse, current_sample_count=None):
        """
        增强的性能检查，考虑数据量因素
        """
        # 小数据模式：完全跳过性能检查
        if current_sample_count is not None and current_sample_count <= self.small_data_threshold:
            logger.info(
                f"小数据保护模式({current_sample_count}条数据)：跳过性能下降检查，"
                f"避免误报（阈值: {self.perf_drop_ratio:.0%}）"
            )
            return False

        # 数据量较少但超过10条：放宽检查条件
        if current_sample_count is not None and current_sample_count <= self.medium_data_threshold:
            adjusted_threshold = self.perf_drop_ratio * 2  # 双倍容忍度
            logger.info(
                f"中等数据保护模式({current_sample_count}条数据)：放宽性能检查，"
                f"调整阈值 {self.perf_drop_ratio:.0%} → {adjusted_threshold:.0%}"
            )
            # 临时调整阈值进行检查
            original_threshold = self.perf_drop_ratio
            self.perf_drop_ratio = adjusted_threshold
            try:
                result = self._original_performance_check(eval_history, baseline_rmse)
                return result
            finally:
                self.perf_drop_ratio = original_threshold

        # 正常数据量：使用原始检查逻辑
        return self._original_performance_check(eval_history, baseline_rmse)

    def _original_performance_check(self, eval_history, baseline_rmse):
        """
        原始性能检查逻辑（分离出来供小数据模式调用）
        """
        if not eval_history or len(eval_history) < self.perf_window + 1:
            return False

        recent_rmse = [h["avg_rmse"] for h in eval_history[-self.perf_window:]]
        recent_avg = sum(recent_rmse) / len(recent_rmse)
        baseline = baseline_rmse or recent_avg

        # 防止除以零
        if baseline == 0:
            return False

        drop_ratio = (recent_avg - baseline) / baseline
        logger.info(
            f"性能检测：基线RMSE={baseline:.4f}，最近{self.perf_window}次平均={recent_avg:.4f}，下降比例={drop_ratio:.2%}"
        )

        trigger = drop_ratio > self.perf_drop_ratio
        if trigger:
            logger.warning(f"性能下降触发：{drop_ratio:.2%} > 阈值{self.perf_drop_ratio:.0%}")

        return trigger

    def _rollback_if_worse(self, eval_result, baseline_rmse, rollback_callback):
        """私有方法：性能差于基线2倍阈值则回滚到上一版本"""
        if eval_result["status"] != "success":
            return

        current_rmse = eval_result["avg_rmse"]
        baseline = baseline_rmse
        if not current_rmse or not baseline:
            return

        drop_ratio = (current_rmse - baseline) / baseline
        if drop_ratio > self.perf_drop_ratio * 2:
            drop_ratio_str = f"{drop_ratio:.2%}" if drop_ratio is not None else "未知"
            threshold_str = f"{(self.perf_drop_ratio * 2):.2%}" if self.perf_drop_ratio is not None else "未知阈值"
            logger.warning(f"性能下降过多（{drop_ratio_str} > {threshold_str}），尝试回滚")

            if rollback_callback:
                rollback_res = rollback_callback(backup_index=-2)
                if rollback_res["success"]:
                    logger.info("回滚成功")
                else:
                    logger.error("回滚失败，需手动干预")

    def _detect_algorithm_type(self, models):
        """检测模型使用的算法类型"""
        if not models:
            return "unknown"

        # 检查第一个模型来判断算法类型
        first_model = next(iter(models.values()))

        # 通过类名判断
        class_name = type(first_model).__name__.lower()
        if 'booster' in class_name and hasattr(first_model, 'num_trees'):
            return "lightgbm"
        elif 'booster' in class_name and hasattr(first_model, 'get_dump'):
            return "xgboost"
        else:
            logger.warning(f"无法识别的模型类型: {class_name}")
            return "unknown"