# hardware_simulator/sensor_emulator.py
import asyncio
import json
import time
import argparse
import numpy as np
import websockets
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExerciseType(Enum):
    SQUAT = "squat"
    PUSHUP = "pushup"
    PLANK = "plank"
    LUNGE = "lunge"
    JUMPING_JACK = "jumping_jack"

@dataclass
class SensorReading:
    timestamp: int
    accelerometer: List[float]
    gyroscope: List[float]
    magnetometer: List[float]

class ExerciseSimulator:
    """运动模拟器基类"""
    
    def __init__(self, exercise_type: ExerciseType, noise_level: float = 0.1):
        self.exercise_type = exercise_type
        self.noise_level = noise_level
        self.phase = 0.0  # 运动阶段 [0, 2π]
        self.rep_count = 0
        self.is_active = False
        
    def add_noise(self, value: float) -> float:
        """添加高斯噪声"""
        return value + np.random.normal(0, self.noise_level)
    
    def generate_reading(self, dt: float) -> SensorReading:
        """生成传感器读数 - 子类实现"""
        raise NotImplementedError

class SquatSimulator(ExerciseSimulator):
    """深蹲模拟器"""
    
    def __init__(self, **kwargs):
        super().__init__(ExerciseType.SQUAT, **kwargs)
        self.squat_depth = 0.8  # 深蹲深度
        self.squat_speed = 1.5  # 深蹲速度
        
    def generate_reading(self, dt: float) -> SensorReading:
        if self.is_active:
            self.phase += dt * self.squat_speed
            if self.phase >= 2 * np.pi:
                self.phase = 0
                self.rep_count += 1
        
        # 模拟深蹲运动的加速度模式
        vertical_movement = np.sin(self.phase) * self.squat_depth
        
        # 加速度 (m/s²)
        acc_x = self.add_noise(0.2 * np.cos(self.phase))  # 前后摇摆
        acc_y = self.add_noise(0.3 * np.sin(self.phase * 0.5))  # 侧向
        acc_z = self.add_noise(9.8 + 2.0 * np.sin(self.phase))  # 垂直主运动
        
        # 陀螺仪 (rad/s)
        gyro_x = self.add_noise(0.1 * np.cos(self.phase))
        gyro_y = self.add_noise(0.15 * np.sin(self.phase))
        gyro_z = self.add_noise(0.05 * np.cos(self.phase * 2))
        
        # 磁力计 (μT)
        mag_x = self.add_noise(22.5 + 2.0 * np.sin(self.phase * 0.1))
        mag_y = self.add_noise(-1.2 + 1.0 * np.cos(self.phase * 0.1))
        mag_z = self.add_noise(48.7 + 0.5 * np.sin(self.phase * 0.1))
        
        return SensorReading(
            timestamp=int(time.time() * 1000),
            accelerometer=[acc_x, acc_y, acc_z],
            gyroscope=[gyro_x, gyro_y, gyro_z],
            magnetometer=[mag_x, mag_y, mag_z]
        )

class PushupSimulator(ExerciseSimulator):
    """俯卧撑模拟器"""
    
    def __init__(self, **kwargs):
        super().__init__(ExerciseType.PUSHUP, **kwargs)
        self.pushup_depth = 0.6
        self.pushup_speed = 1.2
        
    def generate_reading(self, dt: float) -> SensorReading:
        if self.is_active:
            self.phase += dt * self.pushup_speed
            if self.phase >= 2 * np.pi:
                self.phase = 0
                self.rep_count += 1
        
        # 模拟俯卧撑运动模式
        # 加速度
        acc_x = self.add_noise(1.5 * np.sin(self.phase))  # 主要运动方向
        acc_y = self.add_noise(0.2 * np.cos(self.phase))  # 侧向稳定
        acc_z = self.add_noise(9.8 + 0.5 * np.cos(self.phase * 2))  # 重力方向
        
        # 陀螺仪
        gyro_x = self.add_noise(0.2 * np.cos(self.phase))
        gyro_y = self.add_noise(0.1 * np.sin(self.phase * 2))
        gyro_z = self.add_noise(0.05 * np.sin(self.phase))
        
        # 磁力计
        mag_x = self.add_noise(22.5 + 1.0 * np.cos(self.phase * 0.1))
        mag_y = self.add_noise(-1.2 + 0.5 * np.sin(self.phase * 0.1))
        mag_z = self.add_noise(48.7 + 0.3 * np.cos(self.phase * 0.1))
        
        return SensorReading(
            timestamp=int(time.time() * 1000),
            accelerometer=[acc_x, acc_y, acc_z],
            gyroscope=[gyro_x, gyro_y, gyro_z],
            magnetometer=[mag_x, mag_y, mag_z]
        )

class PlankSimulator(ExerciseSimulator):
    """平板支撑模拟器"""
    
    def __init__(self, **kwargs):
        super().__init__(ExerciseType.PLANK, **kwargs)
        self.stability_factor = 0.95  # 稳定性因子
        
    def generate_reading(self, dt: float) -> SensorReading:
        if self.is_active:
            self.phase += dt * 0.5  # 慢速变化
        
        # 模拟静态支撑的微小波动
        stability_noise = (1.0 - self.stability_factor) * 2.0
        
        # 加速度 - 主要是重力和微小的不稳定
        acc_x = self.add_noise(0.1 * np.sin(self.phase) * stability_noise)
        acc_y = self.add_noise(0.05 * np.cos(self.phase * 1.3) * stability_noise)
        acc_z = self.add_noise(9.8 + 0.2 * np.sin(self.phase * 0.7) * stability_noise)
        
        # 陀螺仪 - 微小的调整动作
        gyro_x = self.add_noise(0.02 * np.cos(self.phase * 1.1))
        gyro_y = self.add_noise(0.03 * np.sin(self.phase * 0.9))
        gyro_z = self.add_noise(0.01 * np.cos(self.phase))
        
        # 磁力计 - 基本稳定
        mag_x = self.add_noise(22.5 + 0.1 * np.sin(self.phase * 0.1))
        mag_y = self.add_noise(-1.2 + 0.1 * np.cos(self.phase * 0.1))
        mag_z = self.add_noise(48.7 + 0.1 * np.sin(self.phase * 0.15))
        
        return SensorReading(
            timestamp=int(time.time() * 1000),
            accelerometer=[acc_x, acc_y, acc_z],
            gyroscope=[gyro_x, gyro_y, gyro_z],
            magnetometer=[mag_x, mag_y, mag_z]
        )

class LungeSimulator(ExerciseSimulator):
    """弓步蹲模拟器"""
    
    def __init__(self, **kwargs):
        super().__init__(ExerciseType.LUNGE, **kwargs)
        self.lunge_depth = 0.7
        self.lunge_speed = 1.3
        
    def generate_reading(self, dt: float) -> SensorReading:
        if self.is_active:
            self.phase += dt * self.lunge_speed
            if self.phase >= 2 * np.pi:
                self.phase = 0
                self.rep_count += 1
        
        # 模拟弓步蹲的不对称运动
        # 加速度
        acc_x = self.add_noise(1.2 * np.sin(self.phase))  # 前后移动
        acc_y = self.add_noise(0.8 * np.cos(self.phase) * np.sin(self.phase * 0.5))  # 左右不对称
        acc_z = self.add_noise(9.8 + 1.5 * np.sin(self.phase))  # 垂直运动
        
        # 陀螺仪
        gyro_x = self.add_noise(0.15 * np.cos(self.phase))
        gyro_y = self.add_noise(0.2 * np.sin(self.phase * 1.5))
        gyro_z = self.add_noise(0.1 * np.cos(self.phase * 0.8))
        
        # 磁力计
        mag_x = self.add_noise(22.5 + 1.5 * np.sin(self.phase * 0.1))
        mag_y = self.add_noise(-1.2 + 0.8 * np.cos(self.phase * 0.1))
        mag_z = self.add_noise(48.7 + 0.4 * np.sin(self.phase * 0.1))
        
        return SensorReading(
            timestamp=int(time.time() * 1000),
            accelerometer=[acc_x, acc_y, acc_z],
            gyroscope=[gyro_x, gyro_y, gyro_z],
            magnetometer=[mag_x, mag_y, mag_z]
        )

class JumpingJackSimulator(ExerciseSimulator):
    """开合跳模拟器"""
    
    def __init__(self, **kwargs):
        super().__init__(ExerciseType.JUMPING_JACK, **kwargs)
        self.jump_height = 0.5
        self.jump_speed = 2.0
        
    def generate_reading(self, dt: float) -> SensorReading:
        if self.is_active:
            self.phase += dt * self.jump_speed
            if self.phase >= 2 * np.pi:
                self.phase = 0
                self.rep_count += 1
        
        # 模拟开合跳的高强度运动
        # 加速度
        acc_x = self.add_noise(0.5 * np.cos(self.phase * 2))  # 手臂摆动
        acc_y = self.add_noise(2.0 * np.sin(self.phase))  # 腿部开合
        acc_z = self.add_noise(9.8 + 3.0 * np.sin(self.phase))  # 跳跃垂直运动
        
        # 陀螺仪
        gyro_x = self.add_noise(0.3 * np.sin(self.phase * 2))
        gyro_y = self.add_noise(0.4 * np.cos(self.phase))
        gyro_z = self.add_noise(0.2 * np.sin(self.phase * 1.5))
        
        # 磁力计
        mag_x = self.add_noise(22.5 + 2.0 * np.sin(self.phase * 0.1))
        mag_y = self.add_noise(-1.2 + 1.5 * np.cos(self.phase * 0.1))
        mag_z = self.add_noise(48.7 + 0.8 * np.sin(self.phase * 0.1))
        
        return SensorReading(
            timestamp=int(time.time() * 1000),
            accelerometer=[acc_x, acc_y, acc_z],
            gyroscope=[gyro_x, gyro_y, gyro_z],
            magnetometer=[mag_x, mag_y, mag_z]
        )

class SensorEmulator:
    """传感器模拟器主类"""
    
    def __init__(self, exercise_type: str, user_id: str = "simulator_001"):
        self.user_id = user_id
        self.exercise_type = ExerciseType(exercise_type)
        self.simulator = self._create_simulator()
        self.websocket = None
        self.running = False
        self.sample_rate = 50  # 50Hz采样率
        
    def _create_simulator(self) -> ExerciseSimulator:
        """创建对应的运动模拟器"""
        simulators = {
            ExerciseType.SQUAT: SquatSimulator,
            ExerciseType.PUSHUP: PushupSimulator,
            ExerciseType.PLANK: PlankSimulator,
            ExerciseType.LUNGE: LungeSimulator,
            ExerciseType.JUMPING_JACK: JumpingJackSimulator,
        }
        
        simulator_class = simulators.get(self.exercise_type, SquatSimulator)
        return simulator_class()
    
    async def connect_to_gateway(self, gateway_url: str = "ws://localhost:8000/ws/sensor"):
        """连接到边缘计算网关"""
        try:
            self.websocket = await websockets.connect(gateway_url)
            logger.info(f"Connected to gateway: {gateway_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to gateway: {e}")
            return False
    
    async def start_simulation(self, duration: int = 60):
        """开始模拟传感器数据"""
        if not self.websocket:
            logger.error("Not connected to gateway")
            return
        
        self.running = True
        self.simulator.is_active = True
        
        logger.info(f"Starting {self.exercise_type.value} simulation for {duration} seconds")
        
        start_time = time.time()
        last_time = start_time
        dt = 1.0 / self.sample_rate
        
        try:
            while self.running and (time.time() - start_time) < duration:
                current_time = time.time()
                actual_dt = current_time - last_time
                last_time = current_time
                
                # 生成传感器数据
                sensor_data = self.simulator.generate_reading(actual_dt)
                
                # 添加用户ID
                data_dict = asdict(sensor_data)
                data_dict["user_id"] = self.user_id
                
                # 发送数据
                await self.websocket.send(json.dumps(data_dict))
                
                # 控制采样率
                await asyncio.sleep(dt)
                
                # 每10秒输出状态信息
                if int(current_time - start_time) % 10 == 0 and int(current_time - start_time) > 0:
                    logger.info(f"Simulation running... Reps: {self.simulator.rep_count}, "
                              f"Time: {int(current_time - start_time)}s")
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection to gateway closed")
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self.simulator.is_active = False
            logger.info(f"Simulation completed. Total reps: {self.simulator.rep_count}")
    
    async def stop_simulation(self):
        """停止模拟"""
        self.running = False
        self.simulator.is_active = False
        
        if self.websocket:
            await self.websocket.close()
    
    async def send_rest_data(self, duration: int = 10):
        """发送静息状态数据"""
        if not self.websocket:
            logger.error("Not connected to gateway")
            return
        
        logger.info(f"Sending rest data for {duration} seconds")
        
        start_time = time.time()
        dt = 1.0 / self.sample_rate
        
        try:
            while (time.time() - start_time) < duration:
                # 生成静息状态的传感器数据
                sensor_data = SensorReading(
                    timestamp=int(time.time() * 1000),
                    accelerometer=[
                        np.random.normal(0, 0.05),
                        np.random.normal(0, 0.05),
                        np.random.normal(9.8, 0.1)
                    ],
                    gyroscope=[
                        np.random.normal(0, 0.01),
                        np.random.normal(0, 0.01),
                        np.random.normal(0, 0.01)
                    ],
                    magnetometer=[
                        np.random.normal(22.5, 0.5),
                        np.random.normal(-1.2, 0.3),
                        np.random.normal(48.7, 0.8)
                    ]
                )
                
                data_dict = asdict(sensor_data)
                data_dict["user_id"] = self.user_id
                
                await self.websocket.send(json.dumps(data_dict))
                await asyncio.sleep(dt)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection to gateway closed during rest data")

async def run_simulation_protocol(emulator: SensorEmulator, 
                                  warmup_time: int = 5,
                                  exercise_time: int = 30,
                                  cooldown_time: int = 5):
    """运行完整的模拟协议"""
    
    logger.info("=== Starting Simulation Protocol ===")
    
    # 热身阶段
    logger.info("Phase 1: Warm-up (rest state)")
    await emulator.send_rest_data(warmup_time)
    
    # 运动阶段
    logger.info("Phase 2: Exercise")
    await emulator.start_simulation(exercise_time)
    
    # 冷却阶段
    logger.info("Phase 3: Cool-down (rest state)")
    await emulator.send_rest_data(cooldown_time)
    
    logger.info("=== Simulation Protocol Completed ===")

def main():
    parser = argparse.ArgumentParser(description="EdgeFit-AR Sensor Emulator")
    parser.add_argument("--exercise", "-e", 
                        choices=["squat", "pushup", "plank", "lunge", "jumping_jack"],
                        default="squat",
                        help="Exercise type to simulate")
    parser.add_argument("--duration", "-d", type=int, default=60,
                        help="Simulation duration in seconds")
    parser.add_argument("--gateway", "-g", default="ws://localhost:8000/ws/sensor",
                        help="Gateway WebSocket URL")
    parser.add_argument("--user", "-u", default="simulator_001",
                        help="User ID for simulation")
    parser.add_argument("--protocol", "-p", action="store_true",
                        help="Run full simulation protocol (warmup + exercise + cooldown)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup duration in seconds (protocol mode)")
    parser.add_argument("--cooldown", type=int, default=5,
                        help="Cooldown duration in seconds (protocol mode)")
    
    args = parser.parse_args()
    
    async def run():
        # 创建模拟器
        emulator = SensorEmulator(args.exercise, args.user)
        
        # 连接到网关
        if not await emulator.connect_to_gateway(args.gateway):
            return
        
        try:
            if args.protocol:
                # 运行完整协议
                await run_simulation_protocol(
                    emulator, 
                    args.warmup, 
                    args.duration, 
                    args.cooldown
                )
            else:
                # 只运行运动模拟
                await emulator.start_simulation(args.duration)
        
        finally:
            await emulator.stop_simulation()
    
    # 运行模拟器
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")

if __name__ == "__main__":
    main()