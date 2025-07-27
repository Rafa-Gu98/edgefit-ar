# hardware_simulator/ar_simulator.py
import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Tuple
import websockets
import logging

logger = logging.getLogger(__name__)

class ARSimulator:
    """AR界面模拟器"""
    
    def __init__(self, user_id: str = "ar_user_001"):
        self.user_id = user_id
        self.websocket = None
        self.running = False
        self.current_feedback = None
        
    async def connect_to_gateway(self, gateway_url: str = "ws://localhost:8000/ws/feedback"):
        """连接到AR反馈WebSocket"""
        try:
            self.websocket = await websockets.connect(gateway_url)
            logger.info(f"AR Simulator connected to: {gateway_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect AR simulator: {e}")
            return False
    
    async def listen_for_analysis(self):
        """监听分析结果并生成AR反馈"""
        if not self.websocket:
            return
        
        self.running = True
        logger.info("AR Simulator listening for analysis results...")
        
        try:
            while self.running:
                # 接收分析结果
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "analysis_result":
                    analysis = data.get("data", {})
                    await self._process_analysis_result(analysis)
                elif data.get("type") == "session_event":
                    event = data.get("data", {})
                    await self._process_session_event(event)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("AR Simulator connection closed")
        except Exception as e:
            logger.error(f"Error in AR simulator: {e}")
    
    async def _process_analysis_result(self, analysis: Dict):
        """处理分析结果并生成AR反馈"""
        exercise_type = analysis.get("exercise_type", "unknown")
        form_score = analysis.get("form_score", 0)
        errors = analysis.get("errors", [])
        reps = analysis.get("repetitions", 0)
        
        # 生成AR视觉反馈
        ar_feedback = {
            "type": "ar_feedback",
            "timestamp": int(time.time() * 1000),
            "user_id": self.user_id,
            "visual_elements": self._generate_visual_feedback(exercise_type, form_score, errors),
            "audio_cues": self._generate_audio_cues(errors),
            "rep_counter": reps,
            "form_indicator": self._get_form_indicator_color(form_score)
        }
        
        # 发送反馈给网关
        await self.websocket.send(json.dumps(ar_feedback))
        
        # 控制台输出（模拟AR显示）
        self._display_ar_feedback(ar_feedback)
    
    async def _process_session_event(self, event: Dict):
        """处理会话事件"""
        event_type = event.get("event")
        
        if event_type == "session_started":
            exercise = event.get("exercise_type")
            logger.info(f"🏃 AR: Starting {exercise} session")
            
        elif event_type == "session_stopped":
            summary = event.get("summary", {})
            logger.info(f"🏁 AR: Session completed - {summary}")
    
    def _generate_visual_feedback(self, exercise_type: str, form_score: float, errors: List[Dict]) -> List[Dict]:
        """生成视觉反馈元素"""
        visual_elements = []
        
        # 姿态指示器
        if form_score >= 80:
            visual_elements.append({
                "type": "pose_indicator",
                "color": "green",
                "message": "Perfect form!",
                "position": "center"
            })
        elif form_score >= 60:
            visual_elements.append({
                "type": "pose_indicator", 
                "color": "yellow",
                "message": "Good form",
                "position": "center"
            })
        else:
            visual_elements.append({
                "type": "pose_indicator",
                "color": "red", 
                "message": "Needs improvement",
                "position": "center"
            })
        
        # 错误指示
        for error in errors:
            error_type = error.get("error_type", "")
            suggestion = error.get("suggestion", "")
            
            if error_type == "knee_valgus":
                visual_elements.append({
                    "type": "joint_highlight",
                    "joint": "knees",
                    "color": "red",
                    "animation": "pulse",
                    "message": suggestion
                })
            elif error_type == "forward_lean":
                visual_elements.append({
                    "type": "spine_indicator",
                    "color": "orange",
                    "message": suggestion
                })
        
        return visual_elements
    
    def _generate_audio_cues(self, errors: List[Dict]) -> List[str]:
        """生成音频提示"""
        audio_cues = []
        
        for error in errors:
            error_type = error.get("error_type", "")
            
            if error_type == "knee_valgus":
                audio_cues.append("Keep knees aligned with toes")
            elif error_type == "forward_lean":
                audio_cues.append("Keep chest up and back straight")
            elif error_type == "incomplete_range":
                audio_cues.append("Go deeper for full range of motion")
        
        return audio_cues
    
    def _get_form_indicator_color(self, form_score: float) -> str:
        """获取形态指示器颜色"""
        if form_score >= 80:
            return "green"
        elif form_score >= 60:
            return "yellow"
        else:
            return "red"
    
    def _display_ar_feedback(self, feedback: Dict):
        """控制台显示AR反馈（模拟AR显示）"""
        print("\n" + "="*50)
        print("🥽 AR DISPLAY")
        print("="*50)
        
        # 显示重复次数
        print(f"📊 Reps: {feedback['rep_counter']}")
        
        # 显示形态评分
        color = feedback['form_indicator']
        color_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
        print(f"📈 Form: {color_emoji.get(color, '⚪')} {color.upper()}")
        
        # 显示视觉元素
        for element in feedback['visual_elements']:
            if element['type'] == 'pose_indicator':
                print(f"💡 {element['message']}")
            elif element['type'] == 'joint_highlight':
                print(f"⚠️  {element['joint'].upper()}: {element['message']}")
            elif element['type'] == 'spine_indicator':
                print(f"🏃 POSTURE: {element['message']}")
        
        # 显示音频提示
        for cue in feedback['audio_cues']:
            print(f"🔊 {cue}")
        
        print("="*50)
    
    async def stop(self):
        """停止AR模拟器"""
        self.running = False
        if self.websocket:
            await self.websocket.close()