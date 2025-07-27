# edge_gateway/connection_manager.py
from fastapi import WebSocket
from typing import List
import json
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.feedback_connections: List[WebSocket] = []
        self.health_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """接受传感器数据连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New sensor connection established. Total: {len(self.active_connections)}")
    
    async def connect_feedback(self, websocket: WebSocket):
        """接受AR反馈连接"""
        await websocket.accept()
        self.feedback_connections.append(websocket)
        logger.info(f"New feedback connection established. Total: {len(self.feedback_connections)}")
    
    async def connect_health(self, websocket: WebSocket):
        """接受健康监测连接"""
        await websocket.accept()
        self.health_connections.append(websocket)
        logger.info(f"New health connection established. Total: {len(self.health_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """断开传感器数据连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Sensor connection closed. Remaining: {len(self.active_connections)}")
    
    def disconnect_feedback(self, websocket: WebSocket):
        """断开AR反馈连接"""
        if websocket in self.feedback_connections:
            self.feedback_connections.remove(websocket)
            logger.info(f"Feedback connection closed. Remaining: {len(self.feedback_connections)}")
    
    def disconnect_health(self, websocket: WebSocket):
        """断开健康监测连接"""
        if websocket in self.health_connections:
            self.health_connections.remove(websocket)
            logger.info(f"Health connection closed. Remaining: {len(self.health_connections)}")
    
    async def broadcast_analysis_result(self, result: dict):
        """广播分析结果到所有反馈连接"""
        if self.feedback_connections:
            message = json.dumps({
                "type": "analysis_result",
                "data": result
            })
            
            disconnected = []
            for connection in self.feedback_connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to feedback connection: {e}")
                    disconnected.append(connection)
            
            # 清理断开的连接
            for conn in disconnected:
                self.disconnect_feedback(conn)
    
    async def broadcast_session_event(self, event: dict):
        """广播会话事件到所有连接"""
        message = json.dumps({
            "type": "session_event",
            "data": event
        })
        
        all_connections = self.active_connections + self.feedback_connections
        disconnected = []
        
        for connection in all_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting session event: {e}")
                disconnected.append(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            if conn in self.active_connections:
                self.disconnect(conn)
            elif conn in self.feedback_connections:
                self.disconnect_feedback(conn)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "edge_gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )