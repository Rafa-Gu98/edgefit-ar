# edge_gateway/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import logging
import time
from datetime import datetime
import numpy as np

import rust_engine
from rust_engine import EdgeFitEngine
from model_manager import ModelManager
from data_adapter import DataAdapter
from connection_manager import ConnectionManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EdgeFit-AR Gateway",
    description="边缘计算运动姿态分析网关",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
inference_engine: Optional[EdgeFitEngine] = None
model_manager: Optional[ModelManager] = None
data_adapter: DataAdapter = DataAdapter()
connection_manager: ConnectionManager = ConnectionManager()

# 数据模型
class SensorData(BaseModel):
    timestamp: int
    user_id: str
    accelerometer: List[float]
    gyroscope: List[float]
    magnetometer: List[float]

class PoseKeypoint(BaseModel):
    x: float
    y: float
    confidence: float

class PoseData(BaseModel):
    timestamp: int
    user_id: str
    keypoints: List[PoseKeypoint]

class MultimodalData(BaseModel):
    timestamp: int
    user_id: str
    sensor_data: SensorData
    pose_data: Optional[PoseData] = None

class ExerciseSession(BaseModel):
    exercise_type: str
    user_id: str
    duration_seconds: Optional[int] = None

class AnalysisResult(BaseModel):
    exercise_type: str
    repetitions: int
    form_score: float
    errors: List[Dict]
    calories_burned: float
    muscle_activation: Dict[str, float]
    timestamp: int

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化组件"""
    global inference_engine, model_manager
    
    try:
        # 初始化模型管理器
        model_manager = ModelManager("./models")
        await model_manager.load_models()
        
        # 初始化推理引擎
        inference_engine = EdgeFitEngine("./models")
        
        logger.info("EdgeFit-AR Gateway started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("Shutting down EdgeFit-AR Gateway")

@app.get("/")
async def root():
    """根路径健康检查"""
    return {
        "message": "EdgeFit-AR Gateway",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康状态检查"""
    global inference_engine
    
    status = {
        "gateway": "healthy",
        "inference_engine": "healthy" if inference_engine else "not_initialized",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(connection_manager.active_connections)
    }
    
    return status

@app.post("/api/v1/sensor", response_model=AnalysisResult)
async def process_sensor_data(data: SensorData):
    """处理传感器数据"""
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # 转换数据格式
        sensor_list = data_adapter.sensor_to_list([data])
        
        # 执行推理
        result = inference_engine.process_sensor_data(sensor_list)
        
        # 添加时间戳
        result["timestamp"] = int(time.time() * 1000)
        
        # 广播结果到所有连接
        await connection_manager.broadcast_analysis_result(result)
        
        return AnalysisResult(**result)
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/multimodal", response_model=AnalysisResult)
async def process_multimodal_data(data: MultimodalData):
    """处理多模态数据（传感器+姿态）"""
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # 转换传感器数据
        sensor_list = data_adapter.sensor_to_list([data.sensor_data])
        
        # 如果有姿态数据，融合处理
        if data.pose_data:
            pose_features = data_adapter.pose_to_features(data.pose_data)
            sensor_list = data_adapter.fuse_sensor_pose(sensor_list, pose_features)
        
        # 执行推理
        result = inference_engine.process_sensor_data(sensor_list)
        result["timestamp"] = int(time.time() * 1000)
        
        # 广播结果
        await connection_manager.broadcast_analysis_result(result)
        
        return AnalysisResult(**result)
        
    except Exception as e:
        logger.error(f"Error processing multimodal data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/exercise/start")
async def start_exercise_session(session: ExerciseSession):
    """开始运动会话"""
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        inference_engine.start_exercise_session(session.exercise_type)
        
        # 通知所有连接的客户端
        await connection_manager.broadcast_session_event({
            "event": "session_started",
            "exercise_type": session.exercise_type,
            "user_id": session.user_id,
            "timestamp": int(time.time() * 1000)
        })
        
        return {
            "status": "success",
            "message": f"Started {session.exercise_type} session",
            "timestamp": int(time.time() * 1000)
        }
        
    except Exception as e:
        logger.error(f"Error starting exercise session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/exercise/stop")
async def stop_exercise_session(user_id: str):
    """停止运动会话"""
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        session_summary = inference_engine.stop_exercise_session()
        
        # 通知所有连接的客户端
        await connection_manager.broadcast_session_event({
            "event": "session_stopped",
            "user_id": user_id,
            "summary": session_summary,
            "timestamp": int(time.time() * 1000)
        })
        
        return {
            "status": "success",
            "message": "Exercise session stopped",
            "summary": session_summary,
            "timestamp": int(time.time() * 1000)
        }
        
    except Exception as e:
        logger.error(f"Error stopping exercise session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health_metrics")
async def get_health_metrics():
    """获取健康指标"""
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        metrics = inference_engine.get_health_metrics()
        metrics["timestamp"] = int(time.time() * 1000)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting health metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/supported_exercises")
async def get_supported_exercises():
    """获取支持的运动类型"""
    return {
        "exercises": [
            {
                "id": "squat",
                "name": "深蹲",
                "description": "标准深蹲动作",
                "difficulty": "beginner",
                "muscle_groups": ["quadriceps", "glutes", "hamstrings"]
            },
            {
                "id": "pushup",
                "name": "俯卧撑",
                "description": "标准俯卧撑动作",
                "difficulty": "beginner",
                "muscle_groups": ["pectorals", "triceps", "deltoids"]
            },
            {
                "id": "plank",
                "name": "平板支撑",
                "description": "静态平板支撑",
                "difficulty": "beginner",
                "muscle_groups": ["core", "shoulders", "glutes"]
            },
            {
                "id": "lunge",
                "name": "弓步蹲",
                "description": "前弓步蹲动作",
                "difficulty": "intermediate",
                "muscle_groups": ["quadriceps", "glutes", "calves"]
            },
            {
                "id": "jumping_jack",
                "name": "开合跳",
                "description": "有氧开合跳",
                "difficulty": "beginner",
                "muscle_groups": ["legs", "arms", "core"]
            }
        ]
    }

@app.websocket("/ws/sensor")
async def websocket_sensor_endpoint(websocket: WebSocket):
    """传感器数据WebSocket端点"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # 接收数据
            data = await websocket.receive_text()
            sensor_data = json.loads(data)
            
            # 处理数据
            if inference_engine:
                try:
                    # 转换格式
                    if isinstance(sensor_data, list):
                        sensor_list = data_adapter.dict_list_to_sensor_list(sensor_data)
                    else:
                        sensor_list = data_adapter.dict_to_sensor_list([sensor_data])
                    
                    # 执行推理
                    result = inference_engine.process_sensor_data(sensor_list)
                    result["timestamp"] = int(time.time() * 1000)
                    
                    # 发送结果回客户端
                    await websocket.send_text(json.dumps(result))
                    
                except Exception as e:
                    logger.error(f"Error processing WebSocket sensor data: {e}")
                    await websocket.send_text(json.dumps({
                        "error": str(e),
                        "timestamp": int(time.time() * 1000)
                    }))
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

@app.websocket("/ws/feedback")
async def websocket_feedback_endpoint(websocket: WebSocket):
    """AR反馈WebSocket端点"""
    await connection_manager.connect_feedback(websocket)
    
    try:
        while True:
            # 保持连接，主要用于接收和发送反馈数据
            data = await websocket.receive_text()
            
            # 处理AR反馈数据（如用户交互、手势等）
            feedback_data = json.loads(data)
            
            # 这里可以处理来自AR界面的反馈
            logger.info(f"Received AR feedback: {feedback_data}")
            
            # 可以根据反馈调整推理参数或发送响应
            response = {
                "status": "received",
                "timestamp": int(time.time() * 1000)
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        connection_manager.disconnect_feedback(websocket)
        logger.info("AR feedback WebSocket client disconnected")

@app.websocket("/ws/health")
async def websocket_health_endpoint(websocket: WebSocket):
    """健康监测WebSocket端点"""
    await connection_manager.connect_health(websocket)
    
    try:
        while True:
            if inference_engine:
                # 定期发送健康指标
                health_metrics = inference_engine.get_health_metrics()
                health_metrics["timestamp"] = int(time.time() * 1000)
                
                await websocket.send_text(json.dumps(health_metrics))
            
            # 每秒更新一次
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        connection_manager.disconnect_health(websocket)
        logger.info("Health monitoring WebSocket client disconnected")


@app.websocket("/ws/ar")
async def websocket_ar_endpoint(websocket: WebSocket):
    """AR应用WebSocket端点"""
    
    try:
        # 详细记录连接过程
        logger.info("尝试建立 WebSocket /ws/ar 连接...")
        
        await connection_manager.connect(websocket)
        logger.info("WebSocket /ws/ar 连接成功建立")
        
        while True:
            try:
                # 接收来自AR客户端的数据
                logger.debug("等待接收 WebSocket 数据...")
                data = await websocket.receive_text()
                logger.info(f"收到 AR 数据: {data[:100]}...")  # 只显示前100个字符
                
                # 解析 JSON 数据
                try:
                    ar_data = json.loads(data)
                    logger.debug(f"成功解析 JSON 数据，类型: {type(ar_data)}")
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON 解析失败: {json_err}")
                    error_response = {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": int(time.time() * 1000)
                    }
                    await websocket.send_text(json.dumps(error_response))
                    continue
                
                # 处理AR数据
                if inference_engine:
                    logger.debug("推理引擎可用，开始处理数据")
                    
                    try:
                        # 根据数据类型处理
                        if "sensor_data" in ar_data:
                            logger.debug("处理传感器数据")
                            sensor_list = data_adapter.dict_to_sensor_list([ar_data["sensor_data"]])
                            result = inference_engine.process_sensor_data(sensor_list)
                            result["timestamp"] = int(time.time() * 1000)
                            result["type"] = "analysis_result"
                            
                            response_text = json.dumps(result)
                            await websocket.send_text(response_text)
                            logger.info("成功发送分析结果")
                        
                        elif "pose_data" in ar_data:
                            logger.debug("处理姿态数据")
                            # 处理姿态数据的逻辑
                            response = {
                                "type": "pose_processed",
                                "status": "success",
                                "timestamp": int(time.time() * 1000)
                            }
                            await websocket.send_text(json.dumps(response))
                            logger.info("成功处理姿态数据")
                        
                        else:
                            logger.debug("处理其他类型数据")
                            # 其他类型的AR数据
                            response = {
                                "type": "data_received",
                                "status": "success",
                                "message": "Data received and processed",
                                "timestamp": int(time.time() * 1000)
                            }
                            await websocket.send_text(json.dumps(response))
                            logger.info("成功处理通用数据")
                            
                    except Exception as process_err:
                        logger.error(f"数据处理失败: {process_err}", exc_info=True)
                        error_response = {
                            "type": "error",
                            "message": f"Data processing failed: {str(process_err)}",
                            "timestamp": int(time.time() * 1000)
                        }
                        await websocket.send_text(json.dumps(error_response))
                
                else:
                    logger.warning("推理引擎未初始化")
                    # 推理引擎未初始化
                    error_response = {
                        "type": "error",
                        "message": "Inference engine not initialized",
                        "timestamp": int(time.time() * 1000)
                    }
                    await websocket.send_text(json.dumps(error_response))
                    
            except WebSocketDisconnect:
                logger.info("WebSocket 客户端主动断开连接")
                break
                
            except Exception as msg_err:
                logger.error(f"处理 WebSocket 消息时出错: {msg_err}", exc_info=True)
                try:
                    error_response = {
                        "type": "error",
                        "message": f"Message processing error: {str(msg_err)}",
                        "timestamp": int(time.time() * 1000)
                    }
                    await websocket.send_text(json.dumps(error_response))
                except Exception as send_err:
                    logger.error(f"发送错误消息失败: {send_err}")
                    break
            
    except Exception as conn_err:
        logger.error(f"WebSocket 连接过程中出错: {conn_err}", exc_info=True)
        
    finally:
        try:
            connection_manager.disconnect(websocket)
            logger.info("WebSocket /ws/ar 连接已清理")
        except Exception as cleanup_err:
            logger.error(f"清理 WebSocket 连接时出错: {cleanup_err}")