# hardware_simulator/data_generator.py
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """合成数据生成器"""
    
    def __init__(self, output_dir: str = "./data_engine/datasets/simulated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_exercise_dataset(self, 
                                  exercise_type: str,
                                  num_subjects: int = 20,
                                  reps_per_subject: int = 10,
                                  sample_rate: int = 50) -> str:
        """生成运动数据集"""
        
        logger.info(f"Generating dataset for {exercise_type}...")
        
        all_data = []
        labels = []
        
        for subject_id in range(num_subjects):
            for rep_id in range(reps_per_subject):
                # 生成一次完整的运动数据
                rep_data = self._generate_single_rep(exercise_type, sample_rate)
                
                for sample in rep_data:
                    sample['subject_id'] = subject_id
                    sample['rep_id'] = rep_id
                    sample['exercise_type'] = exercise_type
                    all_data.append(sample)
                    labels.append(exercise_type)
        
        # 保存数据
        dataset_path = self.output_dir / f"{exercise_type}_dataset.json"
        with open(dataset_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        # 保存标签
        labels_path = self.output_dir / f"{exercise_type}_labels.json" 
        with open(labels_path, 'w') as f:
            json.dump(labels, f, indent=2)
        
        logger.info(f"Dataset saved to {dataset_path}")
        return str(dataset_path)
    
    def _generate_single_rep(self, exercise_type: str, sample_rate: int) -> List[Dict]:
        """生成单次重复动作的数据"""
        
        duration = 4.0  # 4秒一次动作
        num_samples = int(duration * sample_rate)
        time_points = np.linspace(0, 2*np.pi, num_samples)
        
        rep_data = []
        
        for i, t in enumerate(time_points):
            timestamp = int(i * (1000 / sample_rate))  # 毫秒时间戳
            
            if exercise_type == "squat":
                acc_data = self._squat_acceleration_pattern(t)
                gyro_data = self._squat_gyroscope_pattern(t)
            elif exercise_type == "pushup":
                acc_data = self._pushup_acceleration_pattern(t)
                gyro_data = self._pushup_gyroscope_pattern(t)
            else:
                acc_data = self._generic_acceleration_pattern(t)
                gyro_data = self._generic_gyroscope_pattern(t)
            
            # 添加噪声
            acc_data = [x + np.random.normal(0, 0.1) for x in acc_data]
            gyro_data = [x + np.random.normal(0, 0.05) for x in gyro_data]
            
            mag_data = [
                22.5 + np.random.normal(0, 1.0),
                -1.2 + np.random.normal(0, 0.5),
                48.7 + np.random.normal(0, 2.0)
            ]
            
            rep_data.append({
                'timestamp': timestamp,
                'accelerometer': acc_data,
                'gyroscope': gyro_data,
                'magnetometer': mag_data
            })
        
        return rep_data
    
    def _squat_acceleration_pattern(self, t: float) -> List[float]:
        """深蹲加速度模式"""
        return [
            0.2 * np.cos(t) + 0.1 * np.sin(2*t),  # X轴
            0.3 * np.sin(t*0.5),  # Y轴  
            9.8 + 2.0 * np.sin(t) + 0.5 * np.cos(2*t)  # Z轴
        ]
    
    def _squat_gyroscope_pattern(self, t: float) -> List[float]:
        """深蹲陀螺仪模式"""
        return [
            0.1 * np.cos(t) + 0.05 * np.sin(3*t),
            0.15 * np.sin(t) + 0.08 * np.cos(1.5*t), 
            0.05 * np.cos(2*t)
        ]
    
    def _pushup_acceleration_pattern(self, t: float) -> List[float]:
        """俯卧撑加速度模式"""
        return [
            1.5 * np.sin(t) + 0.3 * np.cos(2*t),  # 主运动方向
            0.2 * np.cos(t) + 0.1 * np.sin(1.5*t),
            9.8 + 0.5 * np.cos(2*t) + 0.2 * np.sin(t)
        ]
    
    def _pushup_gyroscope_pattern(self, t: float) -> List[float]:
        """俯卧撑陀螺仪模式"""
        return [
            0.2 * np.cos(t) + 0.1 * np.sin(2*t),
            0.1 * np.sin(2*t) + 0.05 * np.cos(3*t),
            0.05 * np.sin(t) + 0.03 * np.cos(1.8*t)
        ]
    
    def _generic_acceleration_pattern(self, t: float) -> List[float]:
        """通用加速度模式"""
        return [
            0.5 * np.sin(t),
            0.3 * np.cos(t),
            9.8 + 1.0 * np.sin(t)
        ]
    
    def _generic_gyroscope_pattern(self, t: float) -> List[float]:
        """通用陀螺仪模式"""
        return [
            0.1 * np.cos(t),
            0.1 * np.sin(t),
            0.05 * np.cos(2*t)
        ]
    
    def generate_error_patterns(self, exercise_type: str, error_types: List[str]) -> str:
        """生成错误动作模式数据"""
        
        logger.info(f"Generating error patterns for {exercise_type}...")
        
        error_data = {}
        
        for error_type in error_types:
            error_samples = []
            
            for _ in range(50):  # 每种错误50个样本
                sample = self._generate_error_sample(exercise_type, error_type)
                error_samples.append(sample)
            
            error_data[error_type] = error_samples
        
        # 保存错误模式数据
        error_path = self.output_dir / f"{exercise_type}_errors.json"
        with open(error_path, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        logger.info(f"Error patterns saved to {error_path}")
        return str(error_path)
    
    def _generate_error_sample(self, exercise_type: str, error_type: str) -> Dict:
        """生成单个错误样本"""
        
        if exercise_type == "squat":
            if error_type == "knee_valgus":
                # 膝盖内扣错误
                return {
                    'accelerometer': [0.5, 0.8, 9.2],  # 异常侧向加速度
                    'gyroscope': [0.2, 0.3, 0.1],
                    'error_severity': np.random.uniform(0.3, 0.9),
                    'description': '膝盖内扣'
                }
            elif error_type == "forward_lean":
                # 前倾错误
                return {
                    'accelerometer': [1.2, 0.2, 8.5],  # 异常前向加速度
                    'gyroscope': [0.3, 0.1, 0.05],
                    'error_severity': np.random.uniform(0.4, 0.8),
                    'description': '身体前倾过度'
                }
        
        elif exercise_type == "pushup":
            if error_type == "incomplete_range":
                # 动作幅度不足
                return {
                    'accelerometer': [0.8, 0.1, 9.9],  # 幅度减小
                    'gyroscope': [0.1, 0.05, 0.02],
                    'error_severity': np.random.uniform(0.2, 0.6),
                    'description': '下降幅度不足'
                }
            elif error_type == "body_sag":
                # 身体下沉
                return {
                    'accelerometer': [1.0, 0.3, 9.2],
                    'gyroscope': [0.15, 0.2, 0.08],
                    'error_severity': np.random.uniform(0.3, 0.7),
                    'description': '身体中段下沉'
                }
        
        # 默认返回通用错误
        return {
            'accelerometer': [0.5, 0.5, 9.5],
            'gyroscope': [0.1, 0.1, 0.05],
            'error_severity': 0.5,
            'description': '通用错误模式'
        }

if __name__ == "__main__":
    # 示例：生成数据集
    generator = SyntheticDataGenerator()
    
    # 生成各种运动的数据集
    exercises = ["squat", "pushup", "plank", "lunge", "jumping_jack"]
    
    for exercise in exercises:
        generator.generate_exercise_dataset(exercise)
        
        # 生成错误模式
        if exercise == "squat":
            error_types = ["knee_valgus", "forward_lean"]
        elif exercise == "pushup":
            error_types = ["incomplete_range", "body_sag"]
        else:
            error_types = ["generic_error"]
        
        generator.generate_error_patterns(exercise, error_types)
    
    print("All datasets generated successfully!")