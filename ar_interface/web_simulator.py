#!/usr/bin/env python3
"""
EdgeFit-AR Web模拟器
在没有Unity的情况下提供AR界面功能
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import json
import asyncio
import websockets
from typing import Dict, List
from pathlib import Path
import uvicorn

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
        self.edge_gateway_url = "ws://localhost:8000/ws/sensor"
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_to_all(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_ar_interface(request: Request):
    """AR界面主页"""
    return templates.TemplateResponse("ar_interface.html", {"request": request})

@app.websocket("/ws/ar")
async def websocket_endpoint(websocket: WebSocket):
    """AR界面WebSocket连接"""
    await manager.connect(websocket)
    try:
        # 连接到边缘网关
        async with websockets.connect(manager.edge_gateway_url) as edge_ws:
            # 创建双向数据流
            async def forward_to_edge():
                async for message in websocket.iter_text():
                    await edge_ws.send(message)
            
            async def forward_from_edge():
                async for message in edge_ws:
                    await websocket.send_text(message)
            
            # 并行处理双向通信
            await asyncio.gather(
                forward_to_edge(),
                forward_from_edge()
            )
    except Exception as e:
        print(f"WebSocket连接错误: {e}")
    finally:
        manager.disconnect(websocket)

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
        }
        
        .control-btn:hover {
            background: rgba(255, 255, 255, 0.3);
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
    </div>

    <script>
        let ws = null;
        let isConnected = false;
        let currentExercise = 'squat';
        let repCount = 0;
        
        // 连接WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/ar`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {
                console.log('WebSocket连接已建立');
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
                isConnected = false;
                updateConnectionStatus(false);
                updatePostureFeedback('连接已断开', 'error');
                
                // 5秒后重连
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket错误:', error);
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
                repCount++;
                document.getElementById('repCount').textContent = repCount;
            }
        }
        
        // 更新运动信息
        function updateExerciseInfo(exercise) {
            document.getElementById('exerciseType').textContent = exercise.name || '未知';
            document.getElementById('exerciseStatus').textContent = exercise.status || '未知';
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
                ws.send(JSON.stringify({
                    type: 'control',
                    action: 'start_exercise',
                    exercise: currentExercise
                }));
                updatePostureFeedback('开始运动！', 'success');
            }
        }
        
        function pauseExercise() {
            if (ws && isConnected) {
                ws.send(JSON.stringify({
                    type: 'control',
                    action: 'pause_exercise'
                }));
                updatePostureFeedback('运动已暂停', 'warning');
            }
        }
        
        function resetExercise() {
            repCount = 0;
            document.getElementById('repCount').textContent = repCount;
            if (ws && isConnected) {
                ws.send(JSON.stringify({
                    type: 'control',
                    action: 'reset_exercise'
                }));
                updatePostureFeedback('已重置', 'success');
            }
        }
        
        // 页面加载完成后连接WebSocket
        window.onload = function() {
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
    print("📱 访问: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)