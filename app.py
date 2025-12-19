"""
煤矿瓦斯风险预测系统 - Flask接口层
作用：提供RESTful API接口，整合核心模型（coal_mine_model.py）与数据库工具（db_utils.py），支持：
  1. 区域措施强度计算
  2. 模型训练（全量/增量自动切换）
  3. 瓦斯指标预测（分源/模型智能切换）
  4. 模型状态查询、重新训练、版本回滚
依赖：Flask、loguru（日志）、项目核心模块
"""
import pandas as pd
from flask import Flask, request, jsonify
import time
import configparser
from performance_monitor import global_monitor
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
# 3. 初始化Flask应用
app = Flask(__name__)
# 4. 初始化数据库工具与核心模型（全局单例，避免重复创建）
db_utils = DBUtils(config_path="config.ini")
model = CoalMineRiskModel(config_path="config.ini")
logger.info(f"Flask应用初始化完成，模型状态：{'已训练' if model.is_trained else '未训练'}")

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
@app.route("/api/model/train", methods=["POST"])
def train_model():
    """
    模型训练接口（支持全量/增量自动切换，事务控制确保数据一致性）
    返回结果：训练状态、样本统计、模型评估详情
    """
    logger.info("接收到【模型训练】请求")
    start_time = time.time()
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
        # 检查是否包含时间相关字段
        temporal_fields = ['measurement_date', 'depth_from_face']
        missing_temporal = []

        for field in temporal_fields:
            if not any(field in d for d in data):
                missing_temporal.append(field)
        if missing_temporal:
            logger.warning(f"训练数据缺少时间相关字段: {missing_temporal}")
            logger.info("时间特征处理可能需要使用默认值或估算值")

            # 为缺失字段添加默认值
            for d in data:
                if 'measurement_date' in missing_temporal:
                    from datetime import datetime
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    d['measurement_date'] = d.get('measurement_date', current_date)

                if 'depth_from_face' in missing_temporal:
                    # 验证孔深度默认设为0（工作面位置）
                    d['depth_from_face'] = d.get('depth_from_face', 0.0)

            logger.info(f"为{len(data)}条样本添加默认时间特征值")
        else:
            logger.info("训练数据包含完整时间特征（测量日期+验证孔深度）")
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

        # ============ 20251218新增：进尺数据验证 ============
        # 检查训练数据中是否包含进尺字段
        if hasattr(model.data_preprocessor, 'enable_cumulative_advance') and \
                model.data_preprocessor.enable_cumulative_advance:
            samples_with_daily_advance = sum(1 for d in data if 'daily_advance' in d)
            total_samples = len(data)
            logger.info(f"进尺数据检查：{samples_with_daily_advance}/{total_samples}条样本包含日进尺数据")
            if samples_with_daily_advance == 0:
                logger.warning("训练数据中未找到daily_advance字段，将使用默认值")
            elif samples_with_daily_advance < total_samples * 0.5:
                logger.warning(
                    f"只有{samples_with_daily_advance}/{total_samples}条样本有日进尺数据，可能影响进尺特征计算")
        # ============ 20251218新增结束 ============
        # Step 2: 自动计算断层影响系数（样本中缺失则补充）
        logger.info(f"计算{len(data)}条样本的断层影响系数")
        data_with_fault = model.calculate_fault_influence_strength(data)
        logger.info(f"断层影响系数计算完成，有效样本数：{len(data_with_fault)}")
        # Step 3: 调用模型训练（事务控制）
        initial_samples = model.total_samples
        logger.info(f"训练前累计样本数：{initial_samples}，本次训练样本数：{len(data_with_fault)}")
        train_result = model.train(data=data_with_fault, epochs=epochs)
        # Step 4: 增强返回结果（补充训练前后对比）
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
            "training_details": {
                "initial_total_samples": initial_samples,
                "current_total_samples": model.total_samples,
                "new_samples_added": model.total_samples - initial_samples,
                "training_mode": training_mode,
                "epochs": epochs,
                "processed_samples": training_stats.get("processed_samples", 0),
                "saved_to_db": training_stats.get("saved_to_db", 0),
                "evaluation_rmse": training_stats.get("evaluation_rmse"),
                "record_id": training_stats.get("record_id")
            }
        }
        duration = time.time() - start_time
        sample_count = len(data) if 'data' in locals() else 0
        global_monitor.record_api_request(
            endpoint="train",
            method="POST",
            duration=duration,
            status_code=200,
            sample_count=sample_count
        )
        # 额外记录训练会话详情
        if 'train_result' in locals() and train_result.get('status') == 'success':
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
        logger.info(f"模型训练完成，状态：{train_result.get('status')}，模式：{training_mode}，新增样本：{enhanced_result['new_samples_added']}")
        return jsonify(enhanced_result)
    except Exception as e:
        # 捕获异常并返回错误详情
        duration = time.time() - start_time
        global_monitor.record_api_request(
            endpoint="train",
            method="POST",
            duration=duration,
            status_code=500,
            sample_count=0
        )
        error_msg = f"模型训练接口处理失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "status": "error",
            "message": error_msg,
            "duration": round(duration, 2),
            "training_details": {
                "input_samples": len(data) if "data" in locals() else 0,
                "error_phase": "training_process",
                "data_rolled_back": True
            }
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
            "predictions": predict_result.get("predictions", [])
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
        return jsonify(simplified_result)
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
            "fixed_eval_set_size": len(model.fixed_evaluation_set) if hasattr(model,'fixed_evaluation_set') and model.fixed_evaluation_set is not None else 0,
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
# app.py - 新增接口：时间特征分析
@app.route("/api/debug/temporal_analysis", methods=["POST"])
def analyze_temporal_features():
    """
    时间特征分析接口
    分析同一位置不同时间测量值的变化规律
    """
    logger.info("接收到【时间特征分析】请求")
    start_time = time.time()
    try:
        # Step 1: 读取数据
        request_data = request.get_json() or {}
        data = request_data.get("data", [])
        if not data:
            return jsonify({
                "success": False,
                "message": "请求数据不能为空",
                "duration": round(time.time() - start_time, 2)
            }), 400
        # Step 2: 转换为DataFrame进行分析
        df = pd.DataFrame(data)
        # Step 3: 检查必要字段
        required_fields = ['x_coord', 'y_coord', 'z_coord', 'measurement_date', 'gas_emission_q']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            return jsonify({
                "success": False,
                "message": f"缺少必要字段: {missing_fields}",
                "missing_fields": missing_fields,
                "duration": round(time.time() - start_time, 2)
            }), 400
        # Step 4: 执行时间特征分析
        analysis_results = {
            "sample_count": len(df),
            "date_range": {
                "min_date": df['measurement_date'].min(),
                "max_date": df['measurement_date'].max(),
                "day_count": (pd.to_datetime(df['measurement_date'].max()) -
                              pd.to_datetime(df['measurement_date'].min())).days
            },
            "spatial_temporal_analysis": [],
            "recommendations": []
        }
        # Step 5: 识别同一位置不同时间的测量
        # 创建位置标识（四舍五入到1米精度）
        df['location_key'] = (
                df['x_coord'].round().astype(str) + '_' +
                df['y_coord'].round().astype(str) + '_' +
                df['z_coord'].round().astype(str)
        )
        # 按位置分组
        location_groups = df.groupby('location_key')

        for location_key, group in location_groups:
            if len(group) > 1:  # 同一位置有多次测量
                # 按时间排序
                group_sorted = group.sort_values('measurement_date')
                # 计算变化
                q_values = group_sorted['gas_emission_q'].tolist()
                dates = group_sorted['measurement_date'].tolist()
                # 计算变化率和趋势
                if len(q_values) >= 2:
                    q_change = q_values[-1] - q_values[0]
                    q_change_percent = (q_change / q_values[0] * 100) if q_values[0] > 0 else 0
                    analysis_results["spatial_temporal_analysis"].append({
                        "location": location_key,
                        "measurement_count": len(group),
                        "date_range": f"{dates[0]} 至 {dates[-1]}",
                        "q_values": q_values,
                        "q_change": round(q_change, 4),
                        "q_change_percent": round(q_change_percent, 2),
                        "dates": dates,
                        "has_significant_change": abs(q_change_percent) > 50  # 变化超过50%视为显著
                    })
        # Step 6: 生成建议
        if analysis_results["spatial_temporal_analysis"]:
            significant_changes = [a for a in analysis_results["spatial_temporal_analysis"]
                                   if a["has_significant_change"]]
            if significant_changes:
                analysis_results["recommendations"].append(
                    f"发现{len(significant_changes)}个位置存在显著时间变化（变化>50%），"
                    f"强烈建议启用时间特征处理"
                )
            else:
                analysis_results["recommendations"].append(
                    "同一位置不同时间测量值变化较小，时间特征可能不是主要影响因素"
                )
        else:
            analysis_results["recommendations"].append(
                "未发现同一位置的多时间测量，时间特征可能不是主要影响因素"
            )
        # Step 7: 返回分析结果
        duration = round(time.time() - start_time, 2)
        logger.info(f"时间特征分析完成，分析{len(df)}条数据，耗时{duration}秒")
        return jsonify({
            "success": True,
            "message": "时间特征分析完成",
            "analysis": analysis_results,
            "duration": duration
        })
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        error_msg = f"时间特征分析失败：{str(e)}"
        logger.error(error_msg, exc_info=True)

        return jsonify({
            "success": False,
            "message": error_msg,
            "duration": duration
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