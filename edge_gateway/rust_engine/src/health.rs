use std::collections::{HashMap, VecDeque};
use crate::SensorData;

pub struct HealthMonitor {
    heart_rate_buffer: VecDeque<f32>,
    calorie_accumulator: f32,
    stress_indicators: VecDeque<f32>,
    fatigue_score: f32,
    recovery_metrics: HashMap<String, f32>,
    activity_intensity: f32,
    step_count: u32,
    last_update: u64,
}

impl HealthMonitor {
    pub fn new() -> Self {
        HealthMonitor {
            heart_rate_buffer: VecDeque::with_capacity(100),
            calorie_accumulator: 0.0,
            stress_indicators: VecDeque::with_capacity(50),
            fatigue_score: 0.0,
            recovery_metrics: HashMap::new(),
            activity_intensity: 0.0,
            step_count: 0,
            last_update: 0,
        }
    }

    pub fn update_metrics(&mut self, sensor_data: &[SensorData]) {
        if sensor_data.is_empty() {
            return;
        }

        for data in sensor_data {
            self.estimate_heart_rate(data);
            self.calculate_stress_indicators(data);
            self.update_fatigue_score(data);
        }
        
        // 更新活动强度
        self.update_activity_intensity(sensor_data);
        
        // 计步
        self.update_step_count(sensor_data);
        
        // 更新恢复指标
        self.update_recovery_metrics();
        
        self.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
    }

    pub fn get_current_metrics(&self) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();
        
        if let Some(&latest_hr) = self.heart_rate_buffer.back() {
            metrics.insert("heart_rate".to_string(), latest_hr);
        }
        
        metrics.insert("calories_burned".to_string(), self.calorie_accumulator);
        metrics.insert("fatigue_score".to_string(), self.fatigue_score);
        metrics.insert("activity_intensity".to_string(), self.activity_intensity);
        metrics.insert("step_count".to_string(), self.step_count as f32);
        
        if let Some(&stress) = self.stress_indicators.back() {
            metrics.insert("stress_level".to_string(), stress);
        }
        
        // 添加恢复指标
        for (key, value) in &self.recovery_metrics {
            metrics.insert(key.clone(), *value);
        }
        
        metrics
    }

    fn estimate_heart_rate(&mut self, data: &SensorData) {
        // 基于加速度数据的简化心率估算
        let magnitude = (data.accelerometer[0].powi(2) + 
                        data.accelerometer[1].powi(2) + 
                        data.accelerometer[2].powi(2)).sqrt();
        
        // 简化的心率估算（实际应该使用更复杂的信号处理）
        let estimated_hr = 60.0 + (magnitude - 9.8).abs() * 10.0;
        let bounded_hr = estimated_hr.max(50.0).min(200.0);
        
        self.heart_rate_buffer.push_back(bounded_hr);
        if self.heart_rate_buffer.len() > 100 {
            self.heart_rate_buffer.pop_front();
        }
    }

    fn calculate_stress_indicators(&mut self, data: &SensorData) {
        // 基于运动变异性计算压力指标
        let accel_variance = data.accelerometer.iter()
            .map(|&x| (x - 9.8).powi(2))
            .sum::<f32>() / 3.0;
        
        let stress_indicator = (accel_variance * 10.0).min(100.0);
        
        self.stress_indicators.push_back(stress_indicator);
        if self.stress_indicators.len() > 50 {
            self.stress_indicators.pop_front();
        }
    }

    fn update_fatigue_score(&mut self, data: &SensorData) {
        // 基于运动强度和持续时间计算疲劳度
        let intensity = (data.accelerometer[0].powi(2) + 
                        data.accelerometer[1].powi(2) + 
                        data.accelerometer[2].powi(2)).sqrt();
        
        let fatigue_increment = (intensity - 9.8).abs() * 0.01;
        self.fatigue_score = (self.fatigue_score + fatigue_increment).min(100.0);
        
        // 自然恢复
        self.fatigue_score *= 0.999;
        
        // 更新卡路里累计
        self.calorie_accumulator += fatigue_increment * 0.5;
    }

    fn update_recovery_metrics(&mut self) {
        // 计算恢复相关指标
        if !self.heart_rate_buffer.is_empty() {
            let avg_hr: f32 = self.heart_rate_buffer.iter().sum::<f32>() / self.heart_rate_buffer.len() as f32;
            let hr_variability = self.calculate_hr_variability();
            
            self.recovery_metrics.insert("avg_heart_rate".to_string(), avg_hr);
            self.recovery_metrics.insert("hr_variability".to_string(), hr_variability);
            
            // 恢复指数（简化计算）
            let recovery_index = 100.0 - (self.fatigue_score * 0.5 + (avg_hr - 70.0).abs() * 0.3);
            self.recovery_metrics.insert("recovery_index".to_string(), recovery_index.max(0.0));
        }
    }

    fn calculate_hr_variability(&self) -> f32 {
        if self.heart_rate_buffer.len() < 2 {
            return 0.0;
        }
        
        let differences: Vec<f32> = self.heart_rate_buffer
            .iter()
            .zip(self.heart_rate_buffer.iter().skip(1))
            .map(|(a, b)| (a - b).abs())
            .collect();
        
        if differences.is_empty() {
            0.0
        } else {
            differences.iter().sum::<f32>() / differences.len() as f32
        }
    }

    fn update_activity_intensity(&mut self, sensor_data: &[SensorData]) {
        // 计算加速度矢量幅值的变化
        let mut intensity_sum = 0.0;
        
        for data in sensor_data {
            let magnitude = (
                data.accelerometer[0].powi(2) + 
                data.accelerometer[1].powi(2) + 
                data.accelerometer[2].powi(2)
            ).sqrt();
            
            // 去除重力影响（约9.8 m/s²）
            let dynamic_acc = (magnitude - 9.8).abs();
            intensity_sum += dynamic_acc;
        }
        
        // 归一化强度值 (0-1)
        self.activity_intensity = (intensity_sum / sensor_data.len() as f32 / 10.0)
            .min(1.0)
            .max(0.0);
    }

    fn update_step_count(&mut self, sensor_data: &[SensorData]) {
        // 简单的计步算法：检测垂直加速度的峰值
        let mut vertical_acc: Vec<f32> = sensor_data.iter()
            .map(|d| d.accelerometer[2]) // Z轴通常是垂直方向
            .collect();
        
        if vertical_acc.len() < 3 {
            return;
        }
        
        // 应用简单的低通滤波
        for i in 1..vertical_acc.len()-1 {
            vertical_acc[i] = (vertical_acc[i-1] + vertical_acc[i] + vertical_acc[i+1]) / 3.0;
        }
        
        // 检测步数（寻找峰值）
        let mut steps = 0;
        let threshold = 1.5; // 步伐检测阈值
        
        for i in 1..vertical_acc.len()-1 {
            if vertical_acc[i] > vertical_acc[i-1] && 
               vertical_acc[i] > vertical_acc[i+1] && 
               vertical_acc[i] > threshold {
                steps += 1;
            }
        }
        
        self.step_count += steps;
    }

    pub fn get_fitness_level(&self) -> String {
        // 基于活动强度和心率变化评估健身水平
        if self.heart_rate_buffer.len() < 10 {
            return "insufficient_data".to_string();
        }
        
        let avg_intensity = self.activity_intensity;
        let hr_variability = self.calculate_hr_variability();
        
        if avg_intensity > 0.7 && hr_variability < 10.0 {
            "excellent".to_string()
        } else if avg_intensity > 0.5 && hr_variability < 15.0 {
            "good".to_string()
        } else if avg_intensity > 0.3 {
            "fair".to_string()
        } else {
            "needs_improvement".to_string()
        }
    }

    pub fn reset_session_metrics(&mut self) {
        self.step_count = 0;
        self.heart_rate_buffer.clear();
        self.activity_intensity = 0.0;
        self.calorie_accumulator = 0.0;
        self.fatigue_score = 0.0;
        self.stress_indicators.clear();
        self.recovery_metrics.clear();
    }
}