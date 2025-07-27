# edge_gateway/model_manager.py
import os
import asyncio
import logging
from typing import Dict, Optional
import torch
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    """模型管理器，负责模型的加载、更新和版本管理"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.model_versions: Dict[str, str] = {}
        
    async def load_models(self):
        """加载所有模型"""
        try:
            # 加载模型配置
            config_path = self.model_path / "model_config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.model_configs = yaml.safe_load(f)
            
            # 加载各个模型
            await self._load_exercise_classifier()
            await self._load_pose_estimator()
            await self._load_error_detector()
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    async def _load_exercise_classifier(self):
        """加载运动分类模型"""
        model_path = self.model_path / "exercise_classifier.pt"
        if model_path.exists():
            try:
                model = torch.jit.load(str(model_path), map_location='cpu')
                model.eval()
                self.loaded_models["exercise_classifier"] = model
                self.model_versions["exercise_classifier"] = "v1.0"
                logger.info("Exercise classifier loaded")
            except Exception as e:
                logger.error(f"Error loading exercise classifier: {e}")
    
    async def _load_pose_estimator(self):
        """加载姿态估计模型"""
        model_path = self.model_path / "pose_estimator.pt"
        if model_path.exists():
            try:
                model = torch.jit.load(str(model_path), map_location='cpu')
                model.eval()
                self.loaded_models["pose_estimator"] = model
                self.model_versions["pose_estimator"] = "v1.0"
                logger.info("Pose estimator loaded")
            except Exception as e:
                logger.error(f"Error loading pose estimator: {e}")
    
    async def _load_error_detector(self):
        """加载错误检测模型"""
        model_path = self.model_path / "error_detector.pt"
        if model_path.exists():
            try:
                model = torch.jit.load(str(model_path), map_location='cpu')
                model.eval()
                self.loaded_models["error_detector"] = model
                self.model_versions["error_detector"] = "v1.0"
                logger.info("Error detector loaded")
            except Exception as e:
                logger.error(f"Error loading error detector: {e}")
    
    def get_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """获取指定模型"""
        return self.loaded_models.get(model_name)
    
    def get_model_version(self, model_name: str) -> Optional[str]:
        """获取模型版本"""
        return self.model_versions.get(model_name)
    
    async def update_model(self, model_name: str, model_path: str):
        """更新指定模型"""
        try:
            new_model = torch.jit.load(model_path, map_location='cpu')
            new_model.eval()
            
            # 备份旧模型
            if model_name in self.loaded_models:
                backup_path = self.model_path / f"{model_name}_backup.pt"
                torch.jit.save(self.loaded_models[model_name], str(backup_path))
            
            # 更新模型
            self.loaded_models[model_name] = new_model
            
            # 更新版本号
            import time
            version = f"v{int(time.time())}"
            self.model_versions[model_name] = version
            
            logger.info(f"Model {model_name} updated to {version}")
            
        except Exception as e:
            logger.error(f"Error updating model {model_name}: {e}")
            raise