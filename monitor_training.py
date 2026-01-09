import requests
import time
import json


def monitor_training_progress():
    """监控训练进度和性能"""

    # 检查配置状态
    config_status = requests.get("http://localhost:5000/api/system/config_status").json()
    print(f"当前配置: {config_status['config_status']['config_file']}")

    # 检查模型状态
    model_status = requests.get("http://localhost:5000/api/model/status").json()
    print(f"累计样本: {model_status['data']['total_samples']}")
    print(f"训练状态: {'已训练' if model_status['data']['is_trained'] else '未训练'}")

    # 检查性能诊断
    perf_diagnosis = requests.get("http://localhost:5000/api/debug/performance_diagnosis").json()
    if perf_diagnosis['success']:
        diagnosis = perf_diagnosis['diagnosis']
        print(f"性能基线: {diagnosis.get('baseline_rmse', '未设置')}")
        print(f"评估历史: {diagnosis.get('eval_history_count', 0)}次")


if __name__ == "__main__":
    monitor_training_progress()