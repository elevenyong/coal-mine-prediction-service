"""
煤矿瓦斯风险预测系统 - Flask接口层
作用：提供RESTful API接口，整合核心模型（coal_mine_model.py）与数据库工具（db_utils.py），支持：
  1. 区域措施强度计算
  2. 模型训练（全量/增量自动切换）
  3. 瓦斯指标预测（分源/模型智能切换）
  4. 模型状态查询、重新训练、版本回滚
依赖：Flask、loguru（日志）、项目核心模块
"""
from datetime import datetime

from flask import Flask, request, jsonify
import time
import configparser
from performance_monitor import global_monitor
from source_prediction import SourcePredictionCalculator

# 导入项目核心模块
from setup_logging import setup_logging
from db_utils import DBUtils
from coal_mine_model import CoalMineRiskModel

# -------------------- 初始化配置 --------------------
# 1. 初始化日志系统（全局唯一）
logger = setup_logging(config_path="config.ini")
# 2. 读取服务器配置（host/port/debug）
config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
server_config = {
    "host": config.get("Server", "host", fallback="0.0.0.0"),
    "port": config.getint("Server", "port", fallback=5000),
    "debug": config.getboolean("Server", "debug", fallback=False)
}
# -------------------- 日志精简开关（训练接口） --------------------
# True ：训练接口只输出关键日志（默认，减少刷屏）
# False：输出详细训练诊断日志（排查数据问题时使用）
BRIEF_TRAINING_LOGS = config.getboolean("Logging", "brief_training_logs", fallback=True)
# 3. 初始化Flask应用
app = Flask(__name__)
# 4. 初始化数据库工具与核心模型（全局单例，避免重复创建）
db_utils = DBUtils(config_path="config.ini")
model = CoalMineRiskModel(config_path="config.ini")
logger.info(f"Flask应用初始化完成，模型状态：{'已训练' if model.is_trained else '未训练'}")
# 初始化分源预测计算器
source_predictor = SourcePredictionCalculator(config_path="config.ini")
# -------------------- 工具方法：将numpy/pandas类型转换为JSON可序列化类型 --------------------
def _to_json_serializable(obj):
    """
    将结果中的 numpy/pandas 标量、NaN、datetime 等转换为 Flask jsonify 可序列化的 Python 原生类型。
    解决：Object of type int64 is not JSON serializable
    """
    # 延迟导入：避免在没有numpy时导致启动失败（项目requirements里一般会有numpy）
    try:
        import numpy as np
    except Exception:
        np = None

    # 1) None / 原生类型
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # 2) datetime
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")

    # 3) dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]

    # 4) numpy 标量 / NaN
    if np is not None:
        # numpy scalar -> python scalar
        if isinstance(obj, np.generic):
            pyv = obj.item()
            # 将 NaN/Inf 规范化（MySQL/JSON都不接受NaN）
            if isinstance(pyv, float) and (np.isnan(pyv) or np.isinf(pyv)):
                return None
            return pyv
        # numpy array -> list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # 兼容 np.nan
        try:
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
        except Exception:
            pass

    # 5) pandas 类型（不强依赖pandas）
    # pandas Timestamp/Timedelta等通常有 to_pydatetime/to_numpy
    if hasattr(obj, "to_pydatetime"):
        try:
            return obj.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    if hasattr(obj, "to_numpy"):
        try:
            return _to_json_serializable(obj.to_numpy())
        except Exception:
            pass

    # 6) 兜底：转字符串（确保接口不崩溃，同时保留信息）
    return str(obj)

# -------------------- 接口1：区域措施强度计算 --------------------
@app.route("/api/model/calculate_regional_strength", methods=["POST"])
def calculate_regional_strength():
    """
    区域措施强度计算接口（基于吨煤钻孔量，参考《煤矿井下瓦斯抽采工程设计规范》）
    请求参数示例（JSON）：
    {
        "drill_total_length": 1200.5,  # 钻孔总长度（米，必选）
        "coal_seam_thickness": 3.2,    # 煤层厚度（米，必选）
        "working_area": 800.0,         # 工作面积（平方米，必选）
        "coal_density": 1.45           # 煤密度（吨/立方米，可选，默认1.4）
    }
    返回结果：包含计算出的区域措施强度（吨煤钻孔量）及输入参数
    """
    logger.info("接收到【区域措施强度计算】请求（吨煤钻孔量逻辑）")
    start_time = time.time()
    try:
        # Step 1: 读取并校验请求数据
        request_data = request.get_json()
        if not request_data:
            logger.warning("请求数据为空")
            return jsonify({
                "status": "error",
                "message": "请求数据不能为空（需为JSON格式）",
                "data": None
            }), 400
        # 校验必选参数（吨煤钻孔量所需字段）
        required_params = ["drill_total_length", "coal_seam_thickness", "working_area"]
        missing_params = [p for p in required_params if p not in request_data]
        if missing_params:
            logger.warning(f"缺少必选参数：{missing_params}")
            return jsonify({
                "status": "error",
                "message": f"缺少必选参数：{','.join(missing_params)}",
                "data": None
            }), 400
        # Step 2: 调用模型计算强度（传递新参数）
        regional_strength = model.calculate_regional_measure_strength([request_data])
        duration = time.time() - start_time
        global_monitor.record_api_request(
            endpoint="calculate_regional_strength",
            method="POST",
            duration=duration,
            status_code=200,
            sample_count=1
        )
        # Step 3: 构建标准化返回结果
        strength_value = regional_strength[0]["strength"] if regional_strength else 0.0
        response = {
            "status": "success",
            "message": "区域措施强度（吨煤钻孔量）计算完成",
            "regional_measure_strength": strength_value,  # 单位：m/t
            "unit": "m/t（吨煤钻孔量）",
            "duration": round(duration, 2),
            "data": {
                "regional_measure_strength": strength_value,
                "input_params": request_data
            }
        }
        logger.info(f"区域措施强度计算成功：{strength_value}m/t，耗时：{duration:.2f}秒")
        return jsonify(response)
    except Exception as e:
        # 捕获异常并返回详细错误信息
        duration = time.time() - start_time
        error_msg = f"区域措施强度计算失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "status": "error",
            "message": error_msg,
            "duration": round(duration, 2),
            "data": None
        }), 500

# -------------------- 接口2：模型训练（全量/增量自动切换） --------------------
# file: app.py (修改训练接口部分)
@app.route("/api/model/train", methods=["POST"])
def train_model():
    """
    模型训练接口（增强时空数据处理）
    """
    logger.info("接收到【模型训练】请求（时空增强版）")
    start_time = time.time()

    # 在函数开头初始化所有变量，避免作用域问题
    epochs = 1  # 默认值
    data = None
    data_with_fault = None
    workface_groups = {}
    dates_list = []
    repeated_coords = {}
    try:
        # Step 1: 读取并校验请求数据
        request_data = request.get_json()
        if not request_data or "data" not in request_data:
            logger.warning("请求缺少'data'字段")
            return jsonify({
                "status": "error",
                "message": "请求数据中必须包含'data'字段（训练样本列表）",
                "training_details": None
            }), 400
        # 校验样本格式（必须为列表）
        data = request_data["data"]
        if not isinstance(data, list):
            logger.warning("'data'字段非列表类型")
            return jsonify({
                "status": "error",
                "message": "'data'字段必须是列表类型（每个元素为样本字典）",
                "training_details": None
            }), 400
        # 校验epochs参数（正整数）
        epochs = request_data.get("epochs", 1)
        if not isinstance(epochs, int) or epochs < 1:
            logger.warning(f"epochs参数无效：{epochs}")
            return jsonify({
                "status": "error",
                "message": "'epochs'必须是大于等于1的整数",
                "training_details": None
            }), 400
        # Step 2: 检查时空数据完整性
        # 精简模式：用一句话概括；详细模式保留原日志
        if BRIEF_TRAINING_LOGS:
            logger.info("开始训练数据校验与增强（时空字段补齐）")
        else:
            logger.info("检查时空数据完整性")
        enhanced_data = []
        for i, sample in enumerate(data):
            # 确保有基本标识
            if 'working_face' not in sample:
                logger.warning(f"样本[{i}]缺少working_face字段，使用默认值")
                sample['working_face'] = f"工作面_{i}"
            # 确保有时间标识
            if 'measurement_date' not in sample:
                # 尝试从其他字段推断
                if 'create_time' in sample:
                    sample['measurement_date'] = sample['create_time'].split(' ')[0]
                else:
                    sample['measurement_date'] = datetime.now().strftime("%Y-%m-%d")
                    logger.warning(f"样本[{i}]缺少measurement_date，使用当前日期")
            # 确保有坐标信息
            if 'x_coord' not in sample or 'y_coord' not in sample:
                logger.warning(f"样本[{i}]缺少坐标信息，可能影响时空分析")
            # 关键：确保有距采面距离
            if 'distance_to_face' not in sample:
                # 尝试计算或估算
                if 'drilling_depth' in sample and 'face_advance_distance' in sample:
                    sample['distance_to_face'] = sample['drilling_depth'] + sample.get('face_advance_distance', 0)
                elif 'distance_from_entrance' in sample:
                    # 简单估算
                    sample['distance_to_face'] = sample['distance_from_entrance'] % 50  # 假设工作面周期为50米
                    logger.warning(f"样本[{i}]估算distance_to_face: {sample['distance_to_face']}")
                else:
                    sample['distance_to_face'] = 0
                    logger.warning(f"样本[{i}]无法确定distance_to_face，设为0")
            enhanced_data.append(sample)
        data = enhanced_data

        # Step 2.5: 训练标签硬校验（防止q、S缺失被预处理中位数填充掩盖）
        # 从配置读取目标字段（默认使用系统配置的两个目标）
        target_str = config.get("Features", "target_features", fallback="drilling_cuttings_s,gas_emission_velocity_q")
        target_features = [x.strip() for x in target_str.split(",") if x.strip()]

        # 缺失率阈值：可在ini中配置（不配则默认10%）
        missing_ratio_threshold = config.getfloat("Model", "target_missing_ratio_threshold", fallback=0.10)

        # 至少需要多少条“标签完整”的样本：可在ini中配置（不配则默认5）
        min_valid_label_samples = config.getint("Model", "min_valid_label_samples", fallback=5)

        if not data:
            return jsonify({
                "status": "error",
                "message": "训练数据为空，无法训练",
                "training_details": {"input_samples": 0}
            }), 400

        # 统计每个目标字段缺失情况
        missing_stats = {}
        valid_label_count = 0  # q、S都非空的样本数（用于硬门槛）
        total_count = len(data)

        for t in target_features:
            missing_count = 0
            for s in data:
                v = s.get(t, None)
                # None/空字符串 视为缺失；数值0允许（不视为缺失）
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    missing_count += 1
            missing_ratio = missing_count / total_count if total_count > 0 else 1.0
            missing_stats[t] = {
                "missing_count": missing_count,
                "total_count": total_count,
                "missing_ratio": round(missing_ratio, 4)
            }

        # 计算“标签完整样本数”：所有目标字段均不缺失
        for s in data:
            ok = True
            for t in target_features:
                v = s.get(t, None)
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    ok = False
                    break
            if ok:
                valid_label_count += 1

        # 硬校验1：任一目标字段缺失率超过阈值 -> 拒绝训练
        high_missing = [t for t, st in missing_stats.items() if st["missing_ratio"] > missing_ratio_threshold]
        if high_missing:
            logger.warning(f"训练标签缺失率过高，拒绝训练：{missing_stats}")
            return jsonify({
                "status": "error",
                "message": (
                    f"训练数据标签缺失率过高，拒绝训练。"
                    f"阈值={missing_ratio_threshold}，超标字段={','.join(high_missing)}"
                ),
                "training_details": {
                    "input_samples": total_count,
                    "valid_label_samples": valid_label_count,
                    "missing_ratio_threshold": missing_ratio_threshold,
                    "missing_stats": missing_stats
                }
            }), 400

        # 硬校验2：标签完整样本数不足 -> 拒绝训练
        if valid_label_count < min_valid_label_samples:
            logger.warning(f"标签完整样本数不足，拒绝训练：valid={valid_label_count}, total={total_count}")
            return jsonify({
                "status": "error",
                "message": (
                    f"标签完整样本数不足，拒绝训练。"
                    f"至少需要{min_valid_label_samples}条标签完整样本（q、S均不缺失）"
                ),
                "training_details": {
                    "input_samples": total_count,
                    "valid_label_samples": valid_label_count,
                    "min_valid_label_samples": min_valid_label_samples,
                    "missing_stats": missing_stats
                }
            }), 400

        # 精简模式：只输出核心结论；详细统计降到DEBUG，避免刷屏
        logger.info(f"训练标签校验通过：total={total_count}, valid_labels={valid_label_count}")
        if BRIEF_TRAINING_LOGS:
            logger.debug(
                f"训练标签校验详情：threshold={missing_ratio_threshold}, missing_stats={missing_stats}"
            )
        else:
            logger.info(
                f"训练标签校验通过：total={total_count}, valid_labels={valid_label_count}, "
                f"threshold={missing_ratio_threshold}, missing_stats={missing_stats}"
            )
        # Step 3: 时空数据统计
        # 精简模式：不逐工作面逐条打印，只保留总览；详细模式保留原日志
        if not BRIEF_TRAINING_LOGS:
            logger.info("时空数据统计:")
        all_dates = []
        if data:
            # 按工作面分组统计
            for sample in data:
                workface = sample.get('working_face', '未知')
                if workface not in workface_groups:
                    workface_groups[workface] = []
                workface_groups[workface].append(sample)
            for workface, samples in workface_groups.items():
                workface_dates = sorted(
                    set(s.get('measurement_date', '') for s in samples if s.get('measurement_date')))
                if workface_dates:
                    all_dates.extend(workface_dates)
                if not BRIEF_TRAINING_LOGS:
                    logger.info(
                        f" 工作面 '{workface}': {len(samples)}条记录，日期范围: {workface_dates[0] if workface_dates else '未知'} 到 {workface_dates[-1] if workface_dates else '未知'}")
                # 检查同一坐标点的重复测量
                coord_count = {}
                for sample in samples:
                    if all(k in sample for k in ['x_coord', 'y_coord', 'z_coord']):
                        coord_key = f"{sample['x_coord']:.1f}_{sample['y_coord']:.1f}_{sample['z_coord']:.1f}"
                        coord_count[coord_key] = coord_count.get(coord_key, 0) + 1
                workface_repeated = {k: v for k, v in coord_count.items() if v > 1}
                if workface_repeated:
                    if not BRIEF_TRAINING_LOGS:
                        logger.info(f"发现{len(workface_repeated)}个坐标点有重复测量")
                    else:
                        logger.debug(f"发现{len(workface_repeated)}个坐标点有重复测量（已汇总）")
                    repeated_coords.update(workface_repeated)
        dates_list = sorted(set(all_dates)) if all_dates else []
        # 精简模式：输出一条总览（工作面数 + 日期范围 + 重复坐标数）
        if BRIEF_TRAINING_LOGS:
            if dates_list:
                logger.info(
                    f"训练数据概览：workfaces={len(workface_groups)}，date_range={dates_list[0]}~{dates_list[-1]}，repeated_coords={len(repeated_coords)}"
                )
            else:
                logger.info(
                    f"训练数据概览：workfaces={len(workface_groups)}，date_range=未知，repeated_coords={len(repeated_coords)}"
                )
        # Step 4: 自动计算断层影响系数
        logger.info(f"计算{len(data)}条样本的断层影响系数")
        data_with_fault = model.calculate_fault_influence_strength(data)
        logger.info(f"断层影响系数计算完成，有效样本数：{len(data_with_fault)}")
        # 调试：检查断层计算后的数据
        if data_with_fault and isinstance(data_with_fault, list):
            if BRIEF_TRAINING_LOGS:
                logger.debug(f"断层计算后数据示例（第一条）：{data_with_fault[0] if data_with_fault else '空'}")
                logger.debug(f"断层计算后数据字段：{list(data_with_fault[0].keys()) if data_with_fault else '空'}")
            else:
                logger.info(f"断层计算后数据示例（第一条）：{data_with_fault[0] if data_with_fault else '空'}")
                logger.info(f"断层计算后数据字段：{list(data_with_fault[0].keys()) if data_with_fault else '空'}")
        # Step 5: 调用模型训练（事务控制）
        initial_samples = model.total_samples
        logger.info(f"开始模型训练：initial_total={initial_samples}，batch={len(data_with_fault)}，epochs={epochs}")
        # 调用模型训练
        train_result = model.train(data=data_with_fault, epochs=epochs)
        # Step 6: 增强返回结果（补充训练前后对比）
        training_stats = train_result.get("training_stats", {})
        training_mode = training_stats.get("training_mode", "未知")
        enhanced_result = {
            "status": train_result.get("status"),
            "message": train_result.get("message"),
            "training_mode": training_mode,
            "new_samples_added": model.total_samples - initial_samples,
            "current_total_samples": model.total_samples,
            "evaluation_rmse": training_stats.get("evaluation_rmse"),
            "duration": round(time.time() - start_time, 2),
            "spatiotemporal_info": {
                "unique_workfaces": len(workface_groups),
                "date_range": f"{dates_list[0] if dates_list else '未知'} 到 {dates_list[-1] if dates_list else '未知'}",
                "repeated_coords_count": len(repeated_coords)
            },
            "training_details": {
                "initial_total_samples": initial_samples,
                "current_total_samples": model.total_samples,
                "new_samples_added": model.total_samples - initial_samples,
                "training_mode": training_mode,
                "epochs": epochs,  # 这里可以安全访问 epochs
                "processed_samples": training_stats.get("processed_samples", 0),
                "saved_to_db": training_stats.get("saved_to_db", 0),
                "evaluation_rmse": training_stats.get("evaluation_rmse"),
                "record_id": training_stats.get("record_id")
            }
        }
        duration = time.time() - start_time
        sample_count = len(data) if data else 0
        global_monitor.record_api_request(
            endpoint="train",
            method="POST",
            duration=duration,
            status_code=200,
            sample_count=sample_count
        )
        # 额外记录训练会话详情
        if train_result.get('status') == 'success':
            training_stats = train_result.get('training_stats', {})
            global_monitor.record_training_session(
                train_mode=training_stats.get('training_mode', 'unknown'),
                sample_count=training_stats.get('processed_samples', 0),
                duration=training_stats.get('training_duration', duration),
                rmse=training_stats.get('evaluation_rmse')
            )
        # 补充评估详情（若有）
        if "evaluation_details" in train_result:
            enhanced_result["evaluation_details"] = train_result["evaluation_details"]
        logger.info(
            f"模型训练完成，状态：{train_result.get('status')}，模式：{training_mode}，新增样本：{enhanced_result['new_samples_added']}")
        return jsonify(_to_json_serializable(enhanced_result))
    except Exception as e:
        # 捕获异常并返回错误详情
        duration = time.time() - start_time
        sample_count = len(data) if data else 0
        global_monitor.record_api_request(
            endpoint="train",
            method="POST",
            duration=duration,
            status_code=500,
            sample_count=sample_count
        )
        # 构建错误信息，避免引用可能不存在的变量
        error_details = {
            "input_samples": sample_count,
            "error_phase": "training_process",
            "data_rolled_back": True,
            "epochs_attempted": epochs if 'epochs' in locals() else 'unknown'  # 安全地引用 epochs
        }
        error_msg = f"模型训练接口处理失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "status": "error",
            "message": error_msg,
            "duration": round(duration, 2),
            "training_details": error_details
        }), 500

# -------------------- 接口3：瓦斯指标预测（分源/模型切换） --------------------
@app.route("/api/model/predict", methods=["POST"])
def predict():
    """
    瓦斯指标预测接口（支持批量预测，分源参数全则用分源结果，否则用模型预测）
    返回结果：每个样本的瓦斯指标预测值（瓦斯涌出量Q、钻屑量S等）
    """
    logger.info("接收到【瓦斯指标预测】请求")
    start_time = time.time()
    try:
        # Step 1: 读取并校验请求数据
        data = request.get_json()
        if not data:
            logger.warning("预测请求数据为空")
            return jsonify({
                "success": False,
                "message": "请求数据不能为空（支持单样本字典或多样本列表）",
                "data": None,
                "duration": round(time.time() - start_time, 2)
            }), 400
        # -------------------- 关键修正：兼容 {"data":[...]} / {"data":{...}} 包装 --------------------
        # 你当前调用方式是 {"data":[{...}]}，若不解包，会把外层dict当成样本，导致缺少 workface_id 等字段
        if isinstance(data, dict) and "data" in data:
            inner = data.get("data")
            if isinstance(inner, list):
                data = inner
            elif isinstance(inner, dict):
                data = [inner]
            else:
                return jsonify({
                    "success": False,
                    "message": "请求体字段 data 必须是对象或数组",
                    "data": None,
                    "duration": round(time.time() - start_time, 2)
                }), 400
        # 统一格式为列表（支持单样本/多样本）
        if not isinstance(data, list):
            data = [data]
        logger.info(f"预测样本数：{len(data)}")
        # Step 2: 自动计算断层影响系数（预测前必须补充）
        logger.info("计算预测样本的断层影响系数")
        data_with_fault = model.calculate_fault_influence_strength(data)
        logger.info(f"断层影响系数计算完成，有效样本数：{len(data_with_fault)}")
        # Step 3: 调用模型执行预测
        predict_result = model.predict(data_with_fault)
        predict_result["duration"] = round(time.time() - start_time, 2)
        # 简化返回结果
        simplified_result = {
            "success": predict_result.get("success", False),
            "message": predict_result.get("message", ""),
            "sample_count": predict_result.get("sample_count", 0),
            "duration": predict_result.get("duration", 0),
            # 防御：失败时 predictions 可能为 None，统一返回空列表，避免下游 len(None) 等二次异常
            "predictions": predict_result.get("predictions") or []
        }
        # 记录预测性能
        duration = time.time() - start_time
        sample_count = len(data) if isinstance(data, list) else 1
        global_monitor.record_api_request(
            endpoint="predict",
            method="POST",
            duration=duration,
            status_code=200,
            sample_count=sample_count
        )
        # 额外记录预测详情
        global_monitor.record_prediction(
            sample_count=sample_count,
            duration=duration,
            success=True
        )
        logger.info(f"瓦斯指标预测完成，成功预测{len(simplified_result['predictions'])}条样本")
        return jsonify(_to_json_serializable(simplified_result))
    except Exception as e:
        # 捕获异常并返回错误详情
        duration = time.time() - start_time
        sample_count = len(data) if 'data' in locals() and isinstance(data, list) else 1
        global_monitor.record_api_request(
            endpoint="predict",
            method="POST",
            duration=duration,
            status_code=500,
            sample_count=sample_count
        )
        global_monitor.record_prediction(
            sample_count=sample_count,
            duration=duration,
            success=False
        )
        error_msg = f"预测接口处理失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "success": False,
            "message": f"服务器错误：{error_msg}",
            "data": None,
            "duration": round(duration, 2),
            "sample_count": len(data) if "data" in locals() else 0
        }), 500

# -------------------- 接口4：模型状态查询 --------------------
@app.route("/api/model/status", methods=["GET"])
def get_model_status():
    """
    模型当前状态查询接口（无需请求参数）
    返回结果：模型训练状态、累计样本数、备份数、最新评估结果等
    """
    logger.info("接收到【模型状态查询】请求")
    start_time = time.time()
    try:
        # Step 1: 调用模型获取状态
        status = model.get_model_status()
        duration = time.time() - start_time
        global_monitor.record_api_request(
            endpoint="status",
            method="GET",
            duration=duration,
            status_code=200,
            sample_count=0
        )
        # Step 2: 构建标准化返回结果
        response = {
            "success": True,
            "message": "模型状态查询成功",
            "is_trained": status["is_trained"],
            "total_samples": status["total_samples"],
            "training_features_count": status["training_features_count"],
            "backup_count": status["backup_count"],
            "algorithm": getattr(model.model_trainer, 'algorithm', 'lightgbm'),  # 新增算法信息
            "duration": round(time.time() - start_time, 2),
            "data": status
        }
        logger.info(f"模型状态查询成功，训练状态：{'已训练' if status['is_trained'] else '未训练'}")
        return jsonify(response)
    except Exception as e:
        # 捕获异常并返回错误
        duration = time.time() - start_time
        global_monitor.record_api_request(
            endpoint="status",
            method="GET",
            duration=duration,
            status_code=500,
            sample_count=0
        )
        error_msg = f"模型状态查询失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "success": False,
            "message": f"服务器错误：{error_msg}",
            "data": None,
            "duration": round(duration, 2)
        }), 500


# -------------------- 接口5：从数据库重新训练模型 --------------------
@app.route("/api/model/retrain", methods=["POST"])
def retrain_model():
    """
    从数据库重新训练模型接口（防止模型被误删除，全量重新训练）
    目的：使用数据库中所有数据重新训练模型，恢复模型状态
    请求参数示例（JSON，可选）：
    {
        "workface_id": 1,        # 可选，筛选特定工作面数据
        "sample_limit": 1000,    # 可选，限制训练样本数（避免内存溢出）
        "force_full_train": true # 可选，强制全量训练（默认true）
    }
    返回结果：重新训练状态、样本统计
    """
    logger.info("接收到【从数据库重新训练】请求（全量重新训练）")
    start_time = time.time()

    try:
        # Step 1: 读取请求参数（可选）
        data = request.get_json() or {}
        workface_id = data.get("workface_id")
        sample_limit = data.get("sample_limit")  # 改名为sample_limit更清晰
        force_full_train = data.get("force_full_train", True)  # 默认强制全量

        # 参数验证
        if sample_limit is not None and (not isinstance(sample_limit, int) or sample_limit < 1):
            logger.warning(f"sample_limit参数无效：{sample_limit}")
            return jsonify({
                "success": False,
                "message": "'sample_limit'必须是大于等于1的整数（或不填）",
                "data": None,
                "duration": round(time.time() - start_time, 2)
            }), 400

        # 记录重新训练前的状态
        before_status = model.get_model_status()
        logger.info(
            f"重新训练前状态：累计样本 {before_status['total_samples']}，训练状态 {'已训练' if before_status['is_trained'] else '未训练'}")

        # Step 2: 调用模型从数据库重新训练（使用全量重新训练方法）
        retrain_result = model.retrain_from_db_full(
            workface_id=workface_id,
            sample_limit=sample_limit,
            force_full_train=force_full_train
        )

        # Step 3: 记录重新训练后的状态
        after_status = model.get_model_status()
        # Step 3.5: 添加时空特征使用说明（新增）
        if retrain_result.get("status") == "success":
            # 检查重新训练使用的数据是否包含时间特征
            training_stats = retrain_result.get("training_stats", {})
            processed_samples = training_stats.get("processed_samples", 0)

            # 记录时空特征使用情况
            retrain_result["temporal_features_used"] = True
            retrain_result["temporal_features_note"] = "重新训练使用了增强的时间特征（掘进天数、距工作面距离、采动影响系数）"

            logger.info(f"重新训练使用时空特征，样本数：{processed_samples}")

        # Step 4: 构建增强的返回结果
        duration = round(time.time() - start_time, 2)
        enhanced_result = {
            "status": retrain_result.get("status", "unknown"),
            "message": retrain_result.get("message", ""),
            "duration": duration,
            "training_details": retrain_result.get("training_stats", {}),
            "model_status_changes": {
                "before_training": {
                    "is_trained": before_status.get("is_trained", False),
                    "total_samples": before_status.get("total_samples", 0),
                    "training_features_count": before_status.get("training_features_count", 0)
                },
                "after_training": {
                    "is_trained": after_status.get("is_trained", False),
                    "total_samples": after_status.get("total_samples", 0),
                    "training_features_count": after_status.get("training_features_count", 0)
                }
            }
        }

        # 补充评估详情（若有）
        if "evaluation_details" in retrain_result:
            enhanced_result["evaluation_details"] = retrain_result["evaluation_details"]

        # 记录监控
        global_monitor.record_api_request(
            endpoint="retrain",
            method="POST",
            duration=duration,
            status_code=200,
            sample_count=retrain_result.get("training_stats", {}).get("processed_samples", 0)
        )

        logger.info(f"从数据库重新训练完成，状态：{retrain_result.get('status')}，耗时：{duration}秒")
        return jsonify(enhanced_result)

    except Exception as e:
        # 捕获异常并返回错误
        duration = round(time.time() - start_time, 2)
        error_msg = f"重新训练接口处理失败：{str(e)}"
        logger.error(error_msg, exc_info=True)

        global_monitor.record_api_request(
            endpoint="retrain",
            method="POST",
            duration=duration,
            status_code=500,
            sample_count=0
        )

        return jsonify({
            "success": False,
            "message": f"服务器错误：{error_msg}",
            "data": None,
            "duration": duration
        }), 500

# -------------------- 接口6：模型回滚到指定备份 --------------------
@app.route("/api/model/rollback", methods=["POST"])
def rollback_model():
    """
    模型回滚接口（回滚到历史备份）
    请求参数示例（JSON，可选）：
    {
        "backup_index": -1  # 备份索引，默认-1（最新备份），-2=上一版，依此类推
    }
    返回结果：回滚状态、备份时间戳
    """
    logger.info("接收到【模型回滚】请求")
    start_time = time.time()

    try:
        # Step 1: 读取请求参数（默认回滚到最新备份）
        data = request.get_json() or {}
        backup_index = data.get("backup_index", -1)
        if not isinstance(backup_index, int):
            logger.warning(f"backup_index参数无效：{backup_index}")
            return jsonify({
                "success": False,
                "message": "'backup_index'必须是整数（如-1=最新备份，-2=上一版）",
                "data": None,
                "duration": round(time.time() - start_time, 2)
            }), 400
        # Step 2: 调用模型执行回滚
        rollback_result = model.rollback_model(backup_index=backup_index)
        rollback_result["duration"] = round(time.time() - start_time, 2)
        logger.info(f"模型回滚完成，状态：{'成功' if rollback_result['success'] else '失败'}")
        return jsonify(rollback_result)
    except Exception as e:
        # 捕获异常并返回错误
        duration = time.time() - start_time
        error_msg = f"模型回滚接口处理失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "success": False,
            "message": f"服务器错误：{error_msg}",
            "data": None,
            "duration": round(duration, 2)
        }), 500

# -------------------- 接口7：动态重载系统配置 --------------------
@app.route("/api/system/reload_config", methods=["POST"])
def reload_system_config():
    """
    动态重载系统配置接口（无需重启服务）
    请求参数示例（JSON，可选）：
    {
        "config_file": "config_phase2.ini",  # 可选，新配置文件路径
        "reload_database": false              # 可选，是否重载数据库配置，默认false
    }
    返回结果：重载状态、配置文件名
    """
    logger.info("接收到【动态重载系统配置】请求")
    start_time = time.time()
    try:
        # Step 1: 读取请求参数
        request_data = request.get_json() or {}
        config_file = request_data.get("config_file")
        reload_database = request_data.get("reload_database", False)
        # Step 2: 记录当前配置状态（用于监控）
        current_status = model.get_model_status()
        current_samples = current_status.get("total_samples", 0)
        current_trained = current_status.get("is_trained", False)

        logger.info(f"配置重载前状态：累计样本 {current_samples}，训练状态 {'已训练' if current_trained else '未训练'}")
        if reload_database:
            logger.warning("用户请求重载数据库配置，将重建数据库连接")
        # Step 3: 调用模型重载配置
        success = model.reload_config(
            new_config_path=config_file,
            reload_database=reload_database
        )
        # Step 4: 构建返回结果
        duration = round(time.time() - start_time, 2)
        if success:
            response = {
                "success": True,
                "message": "系统配置动态重载成功",
                "config_file": config_file or "config.ini",
                "reload_database": reload_database,
                "duration": duration,
                "model_status_after_reload": {
                    "is_trained": model.is_trained,
                    "total_samples": model.total_samples,
                    "training_features_count": len(model.training_features) if model.training_features else 0
                }
            }
            logger.info(f"系统配置动态重载成功，使用配置文件：{config_file or 'config.ini'}")
        else:
            response = {
                "success": False,
                "message": "系统配置动态重载失败",
                "config_file": config_file or "config.ini",
                "reload_database": reload_database,
                "duration": duration
            }
            logger.error(f"系统配置动态重载失败，配置文件：{config_file or 'config.ini'}")
        return jsonify(response)
    except Exception as e:
        # 捕获异常并返回错误
        duration = round(time.time() - start_time, 2)
        error_msg = f"配置重载接口处理失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "success": False,
            "message": error_msg,
            "duration": duration
        }), 500

# -------------------- 接口8：获取当前配置状态 --------------------
@app.route("/api/system/config_status", methods=["GET"])
def get_config_status():
    """
    获取当前系统配置状态接口
    返回结果：当前使用的配置文件、关键参数值
    """
    logger.info("接收到【获取系统配置状态】请求")
    start_time = time.time()
    try:
        # 获取当前关键配置参数
        config_status = {
            "config_file": getattr(model, 'config_path', 'config.ini'),
            "model_params": {
                "full_train_threshold": getattr(model, 'full_train_threshold', 0),
                "min_train_samples": getattr(model, 'min_train_samples', 0),
                "n_estimators": getattr(model.model_trainer, 'n_estimators', 0) if hasattr(model,
                                                                                           'model_trainer') else 0,
                "increment_estimators": getattr(model.model_trainer, 'increment_estimators', 0) if hasattr(model,
                                                                                                           'model_trainer') else 0,
                "num_leaves": getattr(model.model_trainer, 'num_leaves', 0) if hasattr(model, 'model_trainer') else 0,
                "learning_rate": getattr(model.model_trainer, 'learning_rate', 0) if hasattr(model,
                                                                                             'model_trainer') else 0
            },
            "eval_params": {
                "eval_size": getattr(model.model_evaluator, 'eval_size', 0) if hasattr(model, 'model_evaluator') else 0,
                "perf_drop_ratio": getattr(model.model_evaluator, 'perf_drop_ratio', 0) if hasattr(model,
                                                                                                   'model_evaluator') else 0
            }
        }
        duration = round(time.time() - start_time, 2)
        response = {
            "success": True,
            "message": "系统配置状态获取成功",
            "duration": duration,
            "config_status": config_status
        }
        logger.info("系统配置状态获取成功")
        return jsonify(response)
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        error_msg = f"获取配置状态失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "success": False,
            "message": error_msg,
            "duration": duration
        }), 500

# -------------------- 接口9：小数据增量训练 --------------------
@app.route("/api/model/incremental_train", methods=["POST"])
def incremental_train_small_data():
    """
    小数据增量训练专用接口
    放宽性能检查，专为少量数据设计
    请求参数与普通训练接口相同
    """
    logger.info("接收到【小数据增量训练】请求")
    start_time = time.time()
    try:
        # 读取请求数据
        request_data = request.get_json()
        if not request_data or "data" not in request_data:
            return jsonify({
                "status": "error",
                "message": "请求数据中必须包含'data'字段"
            }), 400
        data = request_data["data"]
        if not isinstance(data, list):
            return jsonify({
                "status": "error",
                "message": "'data'字段必须是列表类型"
            }), 400
        # 小数据特殊处理
        if len(data) > 20:
            logger.warning(f"数据量({len(data)})较大，建议使用普通训练接口")
        # 计算断层影响系数
        data_with_fault = model.calculate_fault_influence_strength(data)
        # 临时保存当前配置
        original_perf_drop_ratio = model.model_evaluator.perf_drop_ratio
        try:
            # 临时放宽性能下降阈值
            model.model_evaluator.perf_drop_ratio = 2.0  # 允许200%下降
            # 执行训练
            result = model.train(data=data_with_fault, epochs=1)
            result["small_data_optimized"] = True
            result["performance_check_relaxed"] = True
        finally:
            # 恢复原配置
            model.model_evaluator.perf_drop_ratio = original_perf_drop_ratio
        duration = round(time.time() - start_time, 2)
        result["duration"] = duration
        logger.info(f"小数据增量训练完成: {len(data)}条数据")
        return jsonify(result)
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        error_msg = f"小数据增量训练失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "status": "error",
            "message": error_msg,
            "duration": duration
        }), 500

# -------------------- 接口10：性能诊断接口 --------------------
@app.route("/api/debug/performance_diagnosis", methods=["GET"])
def get_performance_diagnosis():
    """
    性能诊断接口，帮助分析性能下降问题
    """
    logger.info("接收到【性能诊断】请求")
    try:
        diagnosis_info = {
            "baseline_rmse": model.baseline_rmse,
            "eval_history_count": len(model.eval_history),
            "recent_evaluations": model.eval_history[-5:] if model.eval_history else [],
            "perf_drop_ratio": model.model_evaluator.perf_drop_ratio,
            "perf_window": model.model_evaluator.perf_window,
            "total_samples": model.total_samples,
            "is_trained": model.is_trained,
            "has_fixed_eval_set": hasattr(model, 'fixed_evaluation_set') and model.fixed_evaluation_set is not None,
            "fixed_eval_set_size": len(model.fixed_evaluation_set) if hasattr(model,
                                                                              'fixed_evaluation_set') and model.fixed_evaluation_set is not None else 0,
            "training_features_count": len(model.training_features) if model.training_features else 0
        }
        # 获取最近的训练记录
        if hasattr(model, 'training_stats') and model.training_stats:
            diagnosis_info["recent_trainings"] = model.training_stats[-3:]

        return jsonify({
            "success": True,
            "message": "性能诊断信息获取成功",
            "diagnosis": diagnosis_info
        })
    except Exception as e:
        logger.error(f"性能诊断失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"性能诊断失败: {str(e)}"
        }), 500

# -------------------- 接口11：设置固定评估集 --------------------
@app.route("/api/model/set_fixed_evaluation", methods=["POST"])
def set_fixed_evaluation_set():
    """
    手动设置固定评估数据集
    """
    logger.info("接收到【设置固定评估集】请求")
    try:
        request_data = request.get_json() or {}
        evaluation_data = request_data.get("evaluation_data")
        size = request_data.get("size", 50)

        if not evaluation_data:
            return jsonify({
                "success": False,
                "message": "必须提供 evaluation_data 参数"
            }), 400
        fixed_set = model.set_fixed_evaluation_set(evaluation_data, size)
        return jsonify({
            "success": True,
            "message": f"固定评估集设置成功，共{len(fixed_set)}条数据",
            "evaluation_set_size": len(fixed_set)
        })
    except Exception as e:
        logger.error(f"设置固定评估集失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"设置固定评估集失败: {str(e)}"
        }), 500

# -------------------- 新增接口：分源预测法计算瓦斯涌出量 --------------------
@app.route("/api/model/calculate_gas_emission_source", methods=["POST"])
def calculate_gas_emission_source():
    """
    分源预测法独立接口：计算瓦斯涌出量（基于AQ1018-2006标准）
    请求参数示例（JSON，支持单样本/批量）：
    {
        "data": [
            {
                "coal_thickness": 3.2,              # 煤层厚度（m，必选）
                "tunneling_speed": 2.5,             # 掘进速度（m/min，必选）
                "roadway_length": 150.0,            # 巷道长度（m，必选）
                "roadway_cross_section": 12.5,      # 巷道断面积（m²，必选）
                "original_gas_content": 15.8,       # 原始瓦斯含量（m³/t，必选）
                "residual_gas_content": 3.2,        # 残余瓦斯含量（m³/t，必选）
                "coal_density": 1.45,               # 煤密度（t/m³，可选，默认1.4）
                "initial_gas_emission_strength": 0.015  # 初始瓦斯涌出强度（可选）
            }
        ]
    }
    返回结果：包含分源计算瓦斯涌出量及详细分量
    """
    logger.info("接收到【分源预测法计算】请求")
    start_time = time.time()

    try:
        # 读取请求数据
        request_data = request.get_json()
        if not request_data or "data" not in request_data:
            return jsonify({
                "success": False,
                "message": "请求数据中必须包含'data'字段",
                "data": None,
                "duration": round(time.time() - start_time, 2)
            }), 400

        data = request_data["data"]
        if not isinstance(data, list):
            data = [data]  # 支持单样本

        logger.info(f"分源预测计算开始，样本数：{len(data)}")

        # 调用分源预测法计算
        results = source_predictor.calculate_gas_emission_source(data)

        # 统计计算情况
        success_count = sum(1 for r in results if "calculation_error" not in r)
        error_count = len(results) - success_count

        # 计算平均瓦斯涌出量
        valid_results = [r for r in results if "calculation_error" not in r]
        if valid_results:
            avg_total = sum(r["total_gas_emission"] for r in valid_results) / len(valid_results)
        else:
            avg_total = 0.0

        duration = round(time.time() - start_time, 2)

        # 记录性能监控
        global_monitor.record_api_request(
            endpoint="calculate_gas_emission_source",
            method="POST",
            duration=duration,
            status_code=200,
            sample_count=len(data)
        )

        response = {
            "success": True,
            "message": f"分源预测法计算完成，成功{success_count}条，失败{error_count}条",
            "sample_count": len(data),
            "success_count": success_count,
            "error_count": error_count,
            "avg_total_gas_emission": round(avg_total, 4),
            "calculation_method": "分源预测法(AQ1018-2006标准)",
            "duration": duration,
            "results": results
        }

        logger.info(f"分源预测计算完成：平均总瓦斯涌出量={avg_total:.4f}m³/min")
        return jsonify(response)

    except Exception as e:
        duration = round(time.time() - start_time, 2)
        error_msg = f"分源预测法计算失败：{str(e)}"
        logger.error(error_msg, exc_info=True)

        global_monitor.record_api_request(
            endpoint="calculate_gas_emission_source",
            method="POST",
            duration=duration,
            status_code=500,
            sample_count=0
        )

        return jsonify({
            "success": False,
            "message": error_msg,
            "duration": duration,
            "results": None
        }), 500

@app.route("/api/monitoring/performance", methods=["GET"])
def get_performance_metrics():
    """新增接口：获取性能监控指标"""
    try:
        summary = global_monitor.get_performance_summary()
        return jsonify({
            "success": True,
            "message": "性能监控数据获取成功",
            "performance_metrics": summary
        })
    except Exception as e:
        logger.error(f"获取性能监控数据失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取性能监控数据失败: {str(e)}"
        }), 500

@app.route("/api/monitoring/export", methods=["POST"])
def export_monitoring_data():
    """新增接口：导出监控数据"""
    try:
        filename = global_monitor.export_monitoring_data()
        return jsonify({
            "success": True,
            "message": "监控数据导出成功",
            "filename": filename
        })
    except Exception as e:
        logger.error(f"导出监控数据失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"导出监控数据失败: {str(e)}"
        }), 500

@app.route("/api/monitoring/status", methods=["GET"])
def get_monitoring_status():
    """新增接口：获取监控系统状态"""
    try:
        summary = global_monitor.get_performance_summary()
        counts = summary.get('counts', {})

        return jsonify({
            "success": True,
            "message": "监控系统状态获取成功",
            "monitoring_status": {
                "enabled": global_monitor.is_enabled,
                "total_api_requests": counts.get('total_api_requests', 0),
                "total_training_sessions": counts.get('total_training_sessions', 0),
                "total_predictions": counts.get('total_predictions', 0),
                "data_collection_started": True
            }
        })
    except Exception as e:
        logger.error(f"获取监控系统状态失败: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"获取监控系统状态失败: {str(e)}"
        }), 500

# -------------------- 服务启动入口 --------------------
if __name__ == "__main__":
    logger.info("启动煤矿瓦斯风险预测服务")
    logger.info(f"服务地址：{server_config['host']}:{server_config['port']}")
    logger.info(f"调试模式：{'开启' if server_config['debug'] else '关闭'}")

    # 启动Flask服务（生产环境建议用Gunicorn/uWSGI，此处为开发/测试用）
    app.run(
        host=server_config["host"],
        port=server_config["port"],
        debug=server_config["debug"],
        threaded=True
    )