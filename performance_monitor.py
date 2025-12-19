# performance_monitor.py
"""
煤矿瓦斯风险预测系统 - 性能监控模块
新增模块，不修改现有代码结构
"""
import time
import threading
import psutil
from datetime import datetime
from loguru import logger
import json
import os


class PerformanceMonitor:
    """性能监控器 - 新增类，不影响现有代码"""

    def __init__(self, config_path="config.ini"):
        self.monitoring_data = {
            'api_requests': [],
            'training_sessions': [],
            'predictions': [],
            'system_metrics': []
        }
        self.lock = threading.Lock()
        self.is_enabled = True
        self.storage_path = "monitoring_data"
        os.makedirs(self.storage_path, exist_ok=True)

        # 启动后台监控线程
        self._start_background_monitoring()

    def _start_background_monitoring(self):
        """启动后台系统监控"""

        def system_monitor():
            while self.is_enabled:
                try:
                    system_metric = self._collect_system_metric()
                    with self.lock:
                        self.monitoring_data['system_metrics'].append(system_metric)
                        # 保留最近1000条系统指标
                        if len(self.monitoring_data['system_metrics']) > 1000:
                            self.monitoring_data['system_metrics'] = self.monitoring_data['system_metrics'][-1000:]

                    # 每30秒收集一次系统指标
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"系统监控异常: {str(e)}")
                    time.sleep(60)

        monitor_thread = threading.Thread(target=system_monitor, daemon=True)
        monitor_thread.start()
        logger.info("性能监控后台线程已启动")

    def _collect_system_metric(self):
        """收集系统指标"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'process_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }

    def record_api_request(self, endpoint, method, duration, status_code, sample_count=0):
        """记录API请求性能"""
        if not self.is_enabled:
            return

        request_metric = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'duration': duration,
            'status_code': status_code,
            'sample_count': sample_count
        }

        with self.lock:
            self.monitoring_data['api_requests'].append(request_metric)
            # 保留最近2000条API请求记录
            if len(self.monitoring_data['api_requests']) > 2000:
                self.monitoring_data['api_requests'] = self.monitoring_data['api_requests'][-2000:]

    def record_training_session(self, train_mode, sample_count, duration, rmse=None):
        """记录训练会话性能"""
        if not self.is_enabled:
            return

        training_metric = {
            'timestamp': datetime.now().isoformat(),
            'train_mode': train_mode,
            'sample_count': sample_count,
            'duration': duration,
            'rmse': rmse
        }

        with self.lock:
            self.monitoring_data['training_sessions'].append(training_metric)
            # 保留最近500条训练记录
            if len(self.monitoring_data['training_sessions']) > 500:
                self.monitoring_data['training_sessions'] = self.monitoring_data['training_sessions'][-500:]

    def record_prediction(self, sample_count, duration, success=True):
        """记录预测性能"""
        if not self.is_enabled:
            return

        prediction_metric = {
            'timestamp': datetime.now().isoformat(),
            'sample_count': sample_count,
            'duration': duration,
            'success': success
        }

        with self.lock:
            self.monitoring_data['predictions'].append(prediction_metric)
            # 保留最近2000条预测记录
            if len(self.monitoring_data['predictions']) > 2000:
                self.monitoring_data['predictions'] = self.monitoring_data['predictions'][-2000:]

    def get_performance_summary(self):
        """获取性能摘要"""
        with self.lock:
            api_requests = self.monitoring_data['api_requests'][-100:]  # 最近100个请求
            training_sessions = self.monitoring_data['training_sessions'][-50:]  # 最近50次训练
            predictions = self.monitoring_data['predictions'][-100:]  # 最近100次预测
            system_metrics = self.monitoring_data['system_metrics'][-10:]  # 最近10次系统指标

            summary = {
                'timestamp': datetime.now().isoformat(),
                'api_performance': self._calculate_api_performance(api_requests),
                'training_performance': self._calculate_training_performance(training_sessions),
                'prediction_performance': self._calculate_prediction_performance(predictions),
                'system_resources': self._calculate_system_resources(system_metrics),
                'counts': {
                    'total_api_requests': len(self.monitoring_data['api_requests']),
                    'total_training_sessions': len(self.monitoring_data['training_sessions']),
                    'total_predictions': len(self.monitoring_data['predictions'])
                }
            }

            return summary

    def _calculate_api_performance(self, requests):
        """计算API性能指标"""
        if not requests:
            return {}

        durations = [r['duration'] for r in requests]
        status_codes = [r['status_code'] for r in requests]

        return {
            'request_count': len(requests),
            'avg_duration': sum(durations) / len(durations),
            'p95_duration': sorted(durations)[int(len(durations) * 0.95)],
            'max_duration': max(durations),
            'error_rate': sum(1 for code in status_codes if code >= 400) / len(status_codes),
            'endpoint_stats': self._calculate_endpoint_stats(requests)
        }

    def _calculate_endpoint_stats(self, requests):
        """计算端点统计"""
        endpoint_stats = {}
        for req in requests:
            endpoint = req['endpoint']
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {
                    'count': 0,
                    'durations': []
                }
            endpoint_stats[endpoint]['count'] += 1
            endpoint_stats[endpoint]['durations'].append(req['duration'])

        # 计算统计信息
        for endpoint, stats in endpoint_stats.items():
            durations = stats['durations']
            stats['avg_duration'] = sum(durations) / len(durations)
            stats['max_duration'] = max(durations)
            # 清理临时数据
            del stats['durations']

        return endpoint_stats

    def _calculate_training_performance(self, training_sessions):
        """计算训练性能指标"""
        if not training_sessions:
            return {}

        durations = [t['duration'] for t in training_sessions]
        sample_counts = [t['sample_count'] for t in training_sessions]
        rmses = [t['rmse'] for t in training_sessions if t['rmse'] is not None]

        stats = {
            'session_count': len(training_sessions),
            'avg_duration': sum(durations) / len(durations),
            'avg_sample_count': sum(sample_counts) / len(sample_counts),
            'total_samples_processed': sum(sample_counts)
        }

        if rmses:
            stats['avg_rmse'] = sum(rmses) / len(rmses)
            stats['best_rmse'] = min(rmses)

        return stats

    def _calculate_prediction_performance(self, predictions):
        """计算预测性能指标"""
        if not predictions:
            return {}

        durations = [p['duration'] for p in predictions]
        sample_counts = [p['sample_count'] for p in predictions]
        success_count = sum(1 for p in predictions if p['success'])

        return {
            'prediction_count': len(predictions),
            'success_rate': success_count / len(predictions),
            'avg_duration': sum(durations) / len(durations),
            'avg_batch_size': sum(sample_counts) / len(sample_counts),
            'throughput': sum(sample_counts) / sum(durations) if sum(durations) > 0 else 0
        }

    def _calculate_system_resources(self, system_metrics):
        """计算系统资源指标"""
        if not system_metrics:
            return {}

        cpu_values = [m['cpu_percent'] for m in system_metrics]
        memory_values = [m['memory_percent'] for m in system_metrics]

        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values)
        }

    def export_monitoring_data(self):
        """导出监控数据到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.storage_path, f"monitoring_export_{timestamp}.json")

        with self.lock:
            data_to_export = {
                'export_timestamp': datetime.now().isoformat(),
                'monitoring_data': self.monitoring_data
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_export, f, ensure_ascii=False, indent=2)

            logger.info(f"监控数据已导出到: {filename}")
            return filename


# 全局监控器实例
global_monitor = PerformanceMonitor()