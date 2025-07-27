# edge_gateway/data_adapter.py
import numpy as np
from typing import List, Dict, Any
from .main import SensorData, PoseData, PoseKeypoint

class DataAdapter:
    """数据适配器，负责不同格式数据之间的转换"""
    
    def __init__(self):
        self.pose_keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
    
    def sensor_to_list(self, sensor_data_list: List[SensorData]) -> List[Dict[str, Any]]:
        """将SensorData对象转换为字典列表"""
        result = []
        for data in sensor_data_list:
            result.append({
                "timestamp": data.timestamp,
                "accelerometer": data.accelerometer,
                "gyroscope": data.gyroscope,
                "magnetometer": data.magnetometer
            })
        return result
    
    def dict_to_sensor_list(self, dict_list: List[Dict]) -> List[Dict[str, Any]]:
        """将字典列表转换为传感器数据格式"""
        result = []
        for data in dict_list:
            result.append({
                "timestamp": data.get("timestamp", 0),
                "accelerometer": data.get("accelerometer", [0.0, 0.0, 0.0]),
                "gyroscope": data.get("gyroscope", [0.0, 0.0, 0.0]),
                "magnetometer": data.get("magnetometer", [0.0, 0.0, 0.0])
            })
        return result
    
    def dict_list_to_sensor_list(self, dict_list: List[Dict]) -> List[Dict[str, Any]]:
        """处理来自WebSocket的字典数据"""
        return self.dict_to_sensor_list(dict_list)
    
    def pose_to_features(self, pose_data: PoseData) -> np.ndarray:
        """将姿态数据转换为特征向量"""
        features = []
        
        for keypoint in pose_data.keypoints:
            features.extend([keypoint.x, keypoint.y, keypoint.confidence])
        
        # 补齐到17个关键点（如果不足）
        while len(features) < 17 * 3:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features[:17*3])  # 确保只有17个关键点
    
    def fuse_sensor_pose(self, sensor_list: List[Dict], pose_features: np.ndarray) -> List[Dict[str, Any]]:
        """融合传感器数据和姿态特征"""
        # 将姿态特征添加到传感器数据中
        if sensor_list:
            sensor_list[0]["pose_features"] = pose_features.tolist()
        
        return sensor_list
    
    def normalize_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化传感器数据"""
        normalized = sensor_data.copy()
        
        # 加速度数据归一化（假设重力加速度为9.8）
        if "accelerometer" in normalized:
            acc = np.array(normalized["accelerometer"])
            normalized["accelerometer"] = (acc / 9.8).tolist()
        
        # 陀螺仪数据归一化（假设最大角速度为250度/秒）
        if "gyroscope" in normalized:
            gyro = np.array(normalized["gyroscope"])
            normalized["gyroscope"] = (gyro / 250.0).tolist()
        
        # 磁力计数据归一化（假设地磁场强度为50μT）
        if "magnetometer" in normalized:
            mag = np.array(normalized["magnetometer"])
            normalized["magnetometer"] = (mag / 50.0).tolist()
        
        return normalized
    
    def calculate_derived_features(self, sensor_data: List[Dict]) -> Dict[str, Any]:
        """计算衍生特征"""
        if not sensor_data:
            return {}
        
        features = {}
        
        # 计算加速度幅值
        acc_magnitudes = []
        for data in sensor_data:
            acc = np.array(data.get("accelerometer", [0, 0, 0]))
            magnitude = np.linalg.norm(acc)
            acc_magnitudes.append(magnitude)
        
        features["acc_magnitude_mean"] = np.mean(acc_magnitudes)
        features["acc_magnitude_std"] = np.std(acc_magnitudes)
        features["acc_magnitude_max"] = np.max(acc_magnitudes)
        features["acc_magnitude_min"] = np.min(acc_magnitudes)
        
        # 计算陀螺仪角速度幅值
        gyro_magnitudes = []
        for data in sensor_data:
            gyro = np.array(data.get("gyroscope", [0, 0, 0]))
            magnitude = np.linalg.norm(gyro)
            gyro_magnitudes.append(magnitude)
        
        features["gyro_magnitude_mean"] = np.mean(gyro_magnitudes)
        features["gyro_magnitude_std"] = np.std(gyro_magnitudes)
        
        # 计算时间间隔统计
        if len(sensor_data) > 1:
            timestamps = [data.get("timestamp", 0) for data in sensor_data]
            intervals = np.diff(timestamps)
            features["sampling_rate"] = 1000.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0
            features["sampling_jitter"] = np.std(intervals)
        
        return features