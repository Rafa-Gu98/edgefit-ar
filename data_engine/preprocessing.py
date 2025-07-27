# data_engine/preprocessing.py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SensorDataPreprocessor:
    """传感器数据预处理器"""
    
    def __init__(self, sample_rate: int = 50):
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def preprocess_batch(self, sensor_data: List[Dict]) -> np.ndarray:
        """批量预处理传感器数据"""
        
        if not sensor_data:
            return np.array([])
        
        # 转换为numpy数组
        features = []
        for data in sensor_data:
            acc = data.get('accelerometer', [0, 0, 0])
            gyro = data.get('gyroscope', [0, 0, 0])
            mag = data.get('magnetometer', [0, 0, 0])
            features.append(acc + gyro + mag)
        
        features_array = np.array(features)
        
        # 应用预处理步骤
        processed_features = self._apply_preprocessing_pipeline(features_array)
        
        return processed_features
    
    def _apply_preprocessing_pipeline(self, data: np.ndarray) -> np.ndarray:
        """应用预处理流水线"""
        
        # 1. 去除异常值
        data = self._remove_outliers(data)
        
        # 2. 滤波
        data = self._apply_filters(data)
        
        # 3. 标准化
        data = self._normalize_data(data)
        
        # 4. 特征工程
        data = self._extract_engineered_features(data)
        
        return data
    
    def _remove_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """去除异常值"""
        z_scores = np.abs(zscore(data, axis=0))
        outlier_mask = np.any(z_scores > threshold, axis=1)
        
        if np.sum(outlier_mask) > 0:
            logger.warning(f"检测到 {np.sum(outlier_mask)} 个异常值样本")
            # 用中位数替换异常值
            median_values = np.median(data[~outlier_mask], axis=0)
            data[outlier_mask] = median_values
        
        return data
    
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """应用数字滤波器"""
        
        # 低通滤波器去除高频噪声
        nyquist = self.sample_rate / 2
        cutoff = 20  # 20Hz截止频率
        normalized_cutoff = cutoff / nyquist
        
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        
        # 对每个特征维度应用滤波
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据标准化"""
        
        if not self.is_fitted:
            self.scaler.fit(data)
            self.is_fitted = True
        
        return self.scaler.transform(data)
    
    def _extract_engineered_features(self, data: np.ndarray) -> np.ndarray:
        """提取工程特征"""
        
        # 加速度幅值
        acc_magnitude = np.linalg.norm(data[:, :3], axis=1)
        
        # 陀螺仪幅值
        gyro_magnitude = np.linalg.norm(data[:, 3:6], axis=1)
        
        # 磁力计幅值
        mag_magnitude = np.linalg.norm(data[:, 6:9], axis=1)
        
        # 合并原始特征和工程特征
        engineered_features = np.column_stack([
            data,
            acc_magnitude,
            gyro_magnitude, 
            mag_magnitude
        ])
        
        return engineered_features

class DataAugmentation:
    """数据增强器"""
    
    def __init__(self):
        self.augmentation_methods = {
            'noise_injection': self._add_noise,
            'time_warping': self._time_warp,
            'rotation': self._rotate_3d,
            'scaling': self._scale_amplitude,
            'jittering': self._add_jitter
        }
    
    def augment_dataset(self, 
                       data: np.ndarray, 
                       labels: np.ndarray,
                       augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """增强整个数据集"""
        
        augmented_data = [data]
        augmented_labels = [labels]
        
        for _ in range(augmentation_factor):
            aug_method = np.random.choice(list(self.augmentation_methods.keys()))
            aug_data = self.augmentation_methods[aug_method](data.copy())
            
            augmented_data.append(aug_data)
            augmented_labels.append(labels.copy())
        
        return np.vstack(augmented_data), np.hstack(augmented_labels)
    
    def _add_noise(self, data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    def _time_warp(self, data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """时间扭曲"""
        from scipy.interpolate import interp1d
        
        original_length = data.shape[0]
        # 生成扭曲的时间序列
        warp_steps = np.arange(original_length)
        warp_steps = warp_steps + np.random.normal(0, sigma, original_length)
        warp_steps = np.clip(warp_steps, 0, original_length - 1)
        
        # 对每个特征维度进行插值
        warped_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            f = interp1d(np.arange(original_length), data[:, i], 
                        kind='linear', fill_value='extrapolate')
            warped_data[:, i] = f(warp_steps)
        
        return warped_data
    
    def _rotate_3d(self, data: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """3D旋转增强"""
        angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
        
        # 旋转矩阵 (绕Z轴)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0], 
            [0, 0, 1]
        ])
        
        # 分别旋转加速度和陀螺仪数据
        rotated_data = data.copy()
        rotated_data[:, :3] = data[:, :3] @ rotation_matrix.T  # 加速度
        rotated_data[:, 3:6] = data[:, 3:6] @ rotation_matrix.T  # 陀螺仪
        
        return rotated_data
    
    def _scale_amplitude(self, data: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """幅度缩放"""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale_factor
    
    def _add_jitter(self, data: np.ndarray, strength: float = 0.03) -> np.ndarray:
        """添加抖动"""
        jitter = np.random.uniform(-strength, strength, data.shape)
        return data + jitter
