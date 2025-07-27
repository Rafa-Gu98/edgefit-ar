# data_engine/setup_datasets.py
import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, data_root: str = "./data_engine/datasets"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        
        # 创建目录
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_uci_har_dataset(self) -> bool:
        """设置UCI HAR数据集"""
        
        logger.info("设置UCI HAR数据集...")
        
        uci_dir = self.raw_dir / "uci_har"
        uci_dir.mkdir(exist_ok=True)
        
        # 数据集URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = uci_dir / "UCI_HAR_Dataset.zip"
        
        try:
            # 检查是否已下载
            if not zip_path.exists():
                logger.info("下载UCI HAR数据集...")
                self._download_file(url, zip_path)
            
            # 解压数据集
            extract_dir = uci_dir / "UCI HAR Dataset"
            if not extract_dir.exists():
                logger.info("解压UCI HAR数据集...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(uci_dir)
            
            # 处理数据集
            self._process_uci_har_dataset(extract_dir)
            
            logger.info("✓ UCI HAR数据集设置完成")
            return True
            
        except Exception as e:
            logger.error(f"UCI HAR数据集设置失败: {e}")
            return False
    
    def _download_file(self, url: str, destination: Path):
        """下载文件"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    def _process_uci_har_dataset(self, extract_dir: Path):
        """处理UCI HAR数据集"""
        
        # 读取训练数据
        train_x = np.loadtxt(extract_dir / "train" / "X_train.txt")
        train_y = np.loadtxt(extract_dir / "train" / "y_train.txt")
        
        # 读取测试数据
        test_x = np.loadtxt(extract_dir / "test" / "X_test.txt")
        test_y = np.loadtxt(extract_dir / "test" / "y_test.txt")
        
        # 合并数据
        all_x = np.vstack([train_x, test_x])
        all_y = np.hstack([train_y, test_y])
        
        # 活动标签映射
        activity_labels = {
            1: 'WALKING',
            2: 'WALKING_UPSTAIRS', 
            3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING',
            5: 'STANDING',
            6: 'LAYING'
        }
        
        # 保存处理后的数据
        processed_dir = self.processed_dir / "uci_har"
        processed_dir.mkdir(exist_ok=True)
        
        np.save(processed_dir / "features.npy", all_x)
        np.save(processed_dir / "labels.npy", all_y)
        
        # 保存标签映射
        import json
        with open(processed_dir / "label_mapping.json", 'w') as f:
            json.dump(activity_labels, f, indent=2)
        
        logger.info(f"处理完成: {all_x.shape[0]} 个样本, {all_x.shape[1]} 个特征")
    
    def create_synthetic_fitness_dataset(self) -> bool:
        """创建合成健身数据集"""
        
        logger.info("创建合成健身数据集...")
        
        try:
            from hardware_simulator.data_generator import SyntheticDataGenerator
            
            generator = SyntheticDataGenerator(str(self.raw_dir / "synthetic"))
            
            # 生成各种运动的数据集
            exercises = ["squat", "pushup", "plank", "lunge", "jumping_jack"]
            
            all_data = []
            all_labels = []
            
            for i, exercise in enumerate(exercises):
                # 生成数据集
                dataset_path = generator.generate_exercise_dataset(
                    exercise_type=exercise,
                    num_subjects=30,
                    reps_per_subject=15
                )
                
                # 读取生成的数据
                with open(dataset_path, 'r') as f:
                    import json
                    exercise_data = json.load(f)
                
                # 转换为特征向量
                for sample in exercise_data:
                    features = (
                        sample['accelerometer'] + 
                        sample['gyroscope'] + 
                        sample['magnetometer']
                    )
                    all_data.append(features)
                    all_labels.append(i)  # 数字标签
            
            # 转换为numpy数组
            features_array = np.array(all_data)
            labels_array = np.array(all_labels)
            
            # 保存处理后的数据
            synthetic_dir = self.processed_dir / "synthetic_fitness"
            synthetic_dir.mkdir(exist_ok=True)
            
            np.save(synthetic_dir / "features.npy", features_array)
            np.save(synthetic_dir / "labels.npy", labels_array)
            
            # 保存标签映射
            label_mapping = {i: exercise for i, exercise in enumerate(exercises)}
            with open(synthetic_dir / "label_mapping.json", 'w') as f:
                json.dump(label_mapping, f, indent=2)
            
            logger.info(f"✓ 合成数据集创建完成: {features_array.shape[0]} 个样本")
            return True
            
        except Exception as e:
            logger.error(f"合成数据集创建失败: {e}")
            return False
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """获取数据集信息"""
        
        info = {}
        
        for dataset_dir in self.processed_dir.iterdir():
            if dataset_dir.is_dir():
                features_path = dataset_dir / "features.npy"
                labels_path = dataset_dir / "labels.npy"
                mapping_path = dataset_dir / "label_mapping.json"
                
                if all(p.exists() for p in [features_path, labels_path]):
                    features = np.load(features_path)
                    labels = np.load(labels_path)
                    
                    dataset_info = {
                        "samples": features.shape[0],
                        "features": features.shape[1],
                        "classes": len(np.unique(labels)),
                        "path": str(dataset_dir)
                    }
                    
                    if mapping_path.exists():
                        with open(mapping_path, 'r') as f:
                            import json
                            dataset_info["label_mapping"] = json.load(f)
                    
                    info[dataset_dir.name] = dataset_info
        
        return info

def main():
    """主函数 - 设置所有数据集"""
    
    logging.basicConfig(level=logging.INFO)
    
    manager = DatasetManager()
    
    # 设置UCI HAR数据集
    manager.setup_uci_har_dataset()
    
    # 创建合成健身数据集
    manager.create_synthetic_fitness_dataset()
    
    # 显示数据集信息
    info = manager.get_dataset_info()
    
    print("\n" + "="*50)
    print("数据集设置完成")
    print("="*50)
    
    for name, details in info.items():
        print(f"\n📊 {name.upper()}")
        print(f"   样本数: {details['samples']}")
        print(f"   特征数: {details['features']}")
        print(f"   类别数: {details['classes']}")
        if 'label_mapping' in details:
            print(f"   标签: {list(details['label_mapping'].values())}")

if __name__ == "__main__":
    main()