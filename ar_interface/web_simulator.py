#!/usr/bin/env python3
"""
EdgeFit-AR Web模拟器 - 修复版本
在没有Unity的情况下提供AR界面功能
"""

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import List
from pathlib import Path
import uvicorn
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EdgeFit-AR Web Simulator")

# 设置静态文件和模板
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
templates_path = Path(__file__).parent / "templates" 
templates_path.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)

# 连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.edge_gateway_url = "ws://localhost:8000/ws/ar"  # 修改为正确的端点
        self.edge_gateway_connected = False
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"AR客户端连接成功，当前连接数: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"AR客户端断开连接，当前连接数: {len(self.active_connections)}")
    
    async def send_to_all(self, message: dict):
        for connection in self.active_connections[:]:  # 使用切片避免修改列表时的问题
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"发送消息到客户端失败: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_ar_interface(request: Request):
    """AR界面主页"""
    return templates.TemplateResponse("ar_interface.html", {"request": request})

@app.websocket("/ws/ar")
async def websocket_endpoint(websocket: WebSocket):
    """AR界面WebSocket连接 - 直接处理，不转发"""
    await manager.connect(websocket)
    
    try:
        # 发送欢迎消息
        welcome_msg = {
            "type": "system",
            "message": "AR模拟器已连接",
            "timestamp": int(asyncio.get_event_loop().time() * 1000)
        }
        await websocket.send_text(json.dumps(welcome_msg))
        
        # 开始模拟传感器数据
        sensor_task = asyncio.create_task(simulate_sensor_data(websocket))
        
        try:
            while True:
                # 接收来自AR界面的控制消息
                data = await websocket.receive_text()
                logger.info(f"收到AR控制消息: {data}")
                
                try:
                    message = json.loads(data)
                    await handle_ar_control(websocket, message)
                except json.JSONDecodeError:
                    logger.error(f"无效的JSON数据: {data}")
                    
        except WebSocketDisconnect:
            logger.info("AR客户端主动断开连接")
        finally:
            sensor_task.cancel()
            
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}", exc_info=True)
    finally:
        manager.disconnect(websocket)

async def simulate_sensor_data(websocket: WebSocket):
    """模拟传感器数据"""
    import random
    import math
    
    time_step = 0
    exercise_state = "idle"  # idle, active, rep_detected
    rep_count = 0
    
    try:
        while True:
            time_step += 1
            
            # 模拟深蹲动作的传感器数据
            if exercise_state == "active":
                # 模拟深蹲动作的周期性变化
                cycle = math.sin(time_step * 0.1) 
                
                accel_data = [
                    random.uniform(-2, 2) + cycle * 3,  # X轴 - 前后倾斜
                    random.uniform(-1, 1) - abs(cycle) * 8,  # Y轴 - 上下运动
                    random.uniform(-1, 1) + cycle * 1   # Z轴 - 左右摆动
                ]
                
                gyro_data = [
                    random.uniform(-50, 50) + cycle * 30,  # 绕X轴旋转
                    random.uniform(-20, 20),               # 绕Y轴旋转  
                    random.uniform(-30, 30) + cycle * 20   # 绕Z轴旋转
                ]
                
                # 检测是否完成一次深蹲
                if abs(cycle) > 0.95 and exercise_state == "active":
                    rep_count += 1
                    exercise_state = "rep_detected"
                    
                    # 发送重复次数检测结果
                    rep_msg = {
                        "type": "analysis_result",
                        "result": {
                            "rep_detected": True,
                            "feedback": f"完成第{rep_count}次深蹲！",
                            "quality": "good",
                            "rep_count": rep_count
                        },
                        "timestamp": int(asyncio.get_event_loop().time() * 1000)
                    }
                    await websocket.send_text(json.dumps(rep_msg))
                    
                    # 1秒后恢复到active状态
                    await asyncio.sleep(1)
                    exercise_state = "active"
                    
            else:
                # 静止状态的传感器数据
                accel_data = [
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5) - 9.8,  # 重力加速度
                    random.uniform(-0.5, 0.5)
                ]
                
                gyro_data = [
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(-5, 5)
                ]
            
            # 发送传感器数据
            sensor_msg = {
                "type": "sensor_data",
                "data": {
                    "accel": accel_data,
                    "gyro": gyro_data,
                    "timestamp": int(asyncio.get_event_loop().time() * 1000)
                }
            }
            await websocket.send_text(json.dumps(sensor_msg))
            
            # 发送姿态分析结果（每3秒一次）
            if time_step % 30 == 0:
                if exercise_state == "active":
                    feedback_messages = [
                        "保持膝盖与脚尖同向",
                        "下蹲深度很好",
                        "注意保持背部挺直",
                        "动作节奏很棒",
                        "继续保持"
                    ]
                    feedback = random.choice(feedback_messages)
                    quality = random.choice(["good", "excellent"])
                else:
                    feedback = "准备开始运动"
                    quality = "normal"
                
                analysis_msg = {
                    "type": "analysis_result", 
                    "result": {
                        "feedback": feedback,
                        "quality": quality,
                        "rep_detected": False
                    },
                    "timestamp": int(asyncio.get_event_loop().time() * 1000)
                }
                await websocket.send_text(json.dumps(analysis_msg))
            
            await asyncio.sleep(0.1)  # 10Hz更新频率
            
    except asyncio.CancelledError:
        logger.info("传感器数据模拟已停止")
    except Exception as e:
        logger.error(f"传感器数据模拟错误: {e}")

async def handle_ar_control(websocket: WebSocket, message: dict):
    """处理来自AR界面的控制消息"""
    global exercise_state, rep_count
    
    if message.get("type") == "control":
        action = message.get("action")
        
        if action == "start_exercise":
            exercise_state = "active"
            response = {
                "type": "exercise_update",
                "exercise": {
                    "name": "深蹲",
                    "status": "进行中"
                },
                "timestamp": int(asyncio.get_event_loop().time() * 1000)
            }
            await websocket.send_text(json.dumps(response))
            logger.info("开始运动模拟")
            
        elif action == "pause_exercise":
            exercise_state = "idle"
            response = {
                "type": "exercise_update", 
                "exercise": {
                    "name": "深蹲",
                    "status": "已暂停"
                },
                "timestamp": int(asyncio.get_event_loop().time() * 1000)
            }
            await websocket.send_text(json.dumps(response))
            logger.info("暂停运动模拟")
            
        elif action == "reset_exercise":
            exercise_state = "idle"
            rep_count = 0
            response = {
                "type": "exercise_update",
                "exercise": {
                    "name": "深蹲", 
                    "status": "已重置"
                },
                "timestamp": int(asyncio.get_event_loop().time() * 1000)
            }
            await websocket.send_text(json.dumps(response))
            logger.info("重置运动模拟")

# 全局变量
exercise_state = "idle"
rep_count = 0

def create_ar_template():
    """创建AR界面HTML模板"""
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EdgeFit-AR 智能眼镜模拟器</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            overflow: hidden;
        }
        
        .ar-container {
            position: relative;
            width: 100vw;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .glasses-frame {
            width: 800px;
            height: 400px;
            border: 8px solid #333;
            border-radius: 50px;
            background: rgba(0, 0, 0, 0.1);
            position: relative;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        .ar-display {
            width: 100%;
            height: 100%;
            display: flex;
            padding: 20px;
        }
        
        .left-eye, .right-eye {
            flex: 1;
            margin: 0 10px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 20px;
            padding: 20px;
            position: relative;
        }
        
        .exercise-info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 255, 0, 0.2);
            padding: 10px 20px;
            border-radius: 10px;
            border-left: 4px solid #00ff00;
        }
        
        .posture-feedback {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        
        .sensor-data {
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 12px;
        }
        
        .status-indicator {
            position: absolute;
            top: 10px;
            right: 10px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #ff4444;
        }
        
        .status-indicator.connected {
            background: #44ff44;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .error-feedback {
            color: #ff6b6b;
            background: rgba(255, 107, 107, 0.2);
            border-left-color: #ff6b6b;
        }
        
        .warning-feedback {
            color: #ffa726;
            background: rgba(255, 167, 38, 0.2);
            border-left-color: #ffa726;
        }
        
        .success-feedback {
            color: #66bb6a;
            background: rgba(102, 187, 106, 0.2);
            border-left-color: #66bb6a;
        }
        
        .controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
        }
        
        .control-btn {
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.2);
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 25px;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
            border: none;
        }
        
        .control-btn:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        .log-area {
            position: fixed;
            top: 10px;
            left: 10px;
            width: 300px;
            height: 150px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 10px;
            padding: 10px;
            font-family: monospace;
            font-size: 12px;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body>
    <div class="ar-container">
        <div class="glasses-frame">
            <div class="ar-display">
                <div class="left-eye">
                    <div class="status-indicator" id="connectionStatus"></div>
                    <div class="exercise-info" id="exerciseInfo">
                        <div>运动: <span id="exerciseType">深蹲</span></div>
                        <div>次数: <span id="repCount">0</span></div>
                        <div>状态: <span id="exerciseStatus">准备中</span></div>
                    </div>
                    <div class="posture-feedback" id="postureFeedback">
                        等待连接...
                    </div>
                </div>
                
                <div class="right-eye">
                    <div class="sensor-data" id="sensorData">
                        <div>加速度:</div>
                        <div>X: <span id="accelX">0.00</span></div>
                        <div>Y: <span id="accelY">0.00</span></div>
                        <div>Z: <span id="accelZ">0.00</span></div>
                        <div>陀螺仪:</div>
                        <div>X: <span id="gyroX">0.00</span></div>
                        <div>Y: <span id="gyroY">0.00</span></div>
                        <div>Z: <span id="gyroZ">0.00</span></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <button class="control-btn" onclick="startExercise()">开始运动</button>
            <button class="control-btn" onclick="pauseExercise()">暂停</button>
            <button class="control-btn" onclick="resetExercise()">重置</button>
        </div>
        
        <div class="log-area" id="logArea">
            <div>系统日志:</div>
        </div>
    </div>

    <script>
        let ws = null;
        let isConnected = false;
        let currentExercise = 'squat';
        let repCount = 0;
        
        // 添加日志函数
        function addLog(message) {
            const logArea = document.getElementById('logArea');
            const timestamp = new Date().toLocaleTimeString();
            logArea.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            logArea.scrollTop = logArea.scrollHeight;
        }
        
        // 连接WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/ar`;
            
            addLog(`尝试连接: ${wsUrl}`);
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {
                console.log('WebSocket连接已建立');
                addLog('WebSocket连接成功');
                isConnected = true;
                updateConnectionStatus(true);
                updatePostureFeedback('系统已连接', 'success');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleIncomingData(data);
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket连接已关闭');
                addLog(`连接已关闭 (code: ${event.code})`);
                isConnected = false;
                updateConnectionStatus(false);
                updatePostureFeedback('连接已断开', 'error');
                
                // 5秒后重连
                setTimeout(() => {
                    addLog('尝试重新连接...');
                    connectWebSocket();
                }, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket错误:', error);
                addLog(`连接错误: ${error.message || '未知错误'}`);
                updatePostureFeedback('连接错误', 'error');
            };
        }
        
        // 处理接收到的数据
        function handleIncomingData(data) {
            if (data.type === 'sensor_data') {
                updateSensorDisplay(data.data);
            } else if (data.type === 'analysis_result') {
                updatePostureAnalysis(data.result);
            } else if (data.type === 'exercise_update') {
                updateExerciseInfo(data.exercise);
            } else if (data.type === 'system') {
                addLog(`系统: ${data.message}`);
            }
        }
        
        // 更新传感器数据显示
        function updateSensorDisplay(sensorData) {
            if (sensorData.accel) {
                document.getElementById('accelX').textContent = sensorData.accel[0].toFixed(2);
                document.getElementById('accelY').textContent = sensorData.accel[1].toFixed(2);
                document.getElementById('accelZ').textContent = sensorData.accel[2].toFixed(2);
            }
            
            if (sensorData.gyro) {
                document.getElementById('gyroX').textContent = sensorData.gyro[0].toFixed(2);
                document.getElementById('gyroY').textContent = sensorData.gyro[1].toFixed(2);
                document.getElementById('gyroZ').textContent = sensorData.gyro[2].toFixed(2);
            }
        }
        
        // 更新姿态分析结果
        function updatePostureAnalysis(result) {
            const feedback = result.feedback || '正在分析...';
            const quality = result.quality || 'normal';
            
            updatePostureFeedback(feedback, quality);
            
            if (result.rep_detected) {
                repCount = result.rep_count || repCount + 1;
                document.getElementById('repCount').textContent = repCount;
                addLog(`检测到重复动作: ${repCount}`);
            }
        }
        
        // 更新运动信息
        function updateExerciseInfo(exercise) {
            document.getElementById('exerciseType').textContent = exercise.name || '未知';
            document.getElementById('exerciseStatus').textContent = exercise.status || '未知';
            addLog(`运动状态: ${exercise.status}`);
        }
        
        // 更新姿态反馈
        function updatePostureFeedback(message, type) {
            const feedbackElement = document.getElementById('postureFeedback');
            feedbackElement.textContent = message;
            
            // 清除所有类
            feedbackElement.className = 'posture-feedback';
            
            // 添加对应类型的类
            if (type === 'error') {
                feedbackElement.classList.add('error-feedback');
            } else if (type === 'warning') {
                feedbackElement.classList.add('warning-feedback');
            } else if (type === 'success') {
                feedbackElement.classList.add('success-feedback');
            }
        }
        
        // 更新连接状态
        function updateConnectionStatus(connected) {
            const statusElement = document.getElementById('connectionStatus');
            if (connected) {
                statusElement.classList.add('connected');
            } else {
                statusElement.classList.remove('connected');
            }
        }
        
        // 控制函数
        function startExercise() {
            if (ws && isConnected) {
                const msg = {
                    type: 'control',
                    action: 'start_exercise',
                    exercise: currentExercise
                };
                ws.send(JSON.stringify(msg));
                addLog('发送开始运动指令');
                updatePostureFeedback('开始运动！', 'success');
            } else {
                addLog('未连接，无法开始运动');
            }
        }
        
        function pauseExercise() {
            if (ws && isConnected) {
                const msg = {
                    type: 'control',
                    action: 'pause_exercise'
                };
                ws.send(JSON.stringify(msg));
                addLog('发送暂停指令');
                updatePostureFeedback('运动已暂停', 'warning');
            } else {
                addLog('未连接，无法暂停运动');
            }
        }
        
        function resetExercise() {
            repCount = 0;
            document.getElementById('repCount').textContent = repCount;
            if (ws && isConnected) {
                const msg = {
                    type: 'control',
                    action: 'reset_exercise'
                };
                ws.send(JSON.stringify(msg));
                addLog('发送重置指令');
                updatePostureFeedback('已重置', 'success');
            } else {
                addLog('未连接，仅本地重置');
            }
        }
        
        // 页面加载完成后连接WebSocket
        window.onload = function() {
            addLog('页面加载完成，初始化连接...');
            connectWebSocket();
        };
    </script>
</body>
</html>'''
    
    # 创建模板目录和文件
    template_file = templates_path / "ar_interface.html"
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ AR界面模板已创建: {template_file}")

if __name__ == "__main__":
    # 创建HTML模板
    create_ar_template()
    
    print("🚀 启动EdgeFit-AR Web模拟器...")
    print("📱 访问: http://localhost:8002")
    print("📡 不再依赖边缘网关，使用内置传感器模拟")
    uvicorn.run(app, host="0.0.0.0", port=8002)