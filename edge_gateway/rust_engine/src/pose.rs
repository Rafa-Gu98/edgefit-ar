// edge_gateway/rust_engine/src/pose.rs
use crate::SensorData;

#[derive(Debug, Clone)]
pub struct PoseAnalysis {
    pub form_score: f32,
    pub knee_angle: Option<f32>,
    pub elbow_angle: Option<f32>,
    pub back_angle: Option<f32>,
    pub detected_repetition: Option<bool>,
    pub movement_phase: MovementPhase,
}

#[derive(Debug, Clone)]
pub enum MovementPhase {
    Rest,
    Eccentric,  // 下降阶段
    Bottom,     // 最低点
    Concentric, // 上升阶段
}

pub struct PoseAnalyzer {
    previous_positions: Vec<(f32, f32, f32)>,
    movement_state: MovementPhase,
    rep_threshold: f32,
    smoothing_window: Vec<f32>,
}

impl PoseAnalyzer {
    pub fn new() -> Self {
        PoseAnalyzer {
            previous_positions: Vec::new(),
            movement_state: MovementPhase::Rest,
            rep_threshold: 0.3,
            smoothing_window: Vec::new(),
        }
    }

    pub fn analyze_pose(&mut self, features: &[f32], exercise_type: &str) -> Result<PoseAnalysis, String> {
        match exercise_type {
            "squat" => self.analyze_squat(features),
            "pushup" => self.analyze_pushup(features),
            "plank" => self.analyze_plank(features),
            _ => self.analyze_generic(features),
        }
    }

    fn analyze_squat(&mut self, features: &[f32]) -> Result<PoseAnalysis, String> {
        // 从特征中提取关键信息
        let vertical_acceleration = features.get(2).unwrap_or(&0.0);
        let knee_flex_estimate = self.estimate_knee_flexion(features);
        
        // 平滑处理
        self.smoothing_window.push(*vertical_acceleration);
        if self.smoothing_window.len() > 10 {
            self.smoothing_window.remove(0);
        }
        
        let smoothed_accel: f32 = self.smoothing_window.iter().sum::<f32>() / self.smoothing_window.len() as f32;
        
        // 检测动作阶段
        let (new_phase, rep_detected) = self.detect_squat_phase(smoothed_accel);
        let old_phase = std::mem::replace(&mut self.movement_state, new_phase);
        
        // 计算形态评分
        let form_score = self.calculate_squat_form_score(knee_flex_estimate, features);
        
        Ok(PoseAnalysis {
            form_score,
            knee_angle: Some(knee_flex_estimate),
            elbow_angle: None,
            back_angle: self.estimate_back_angle(features),
            detected_repetition: Some(rep_detected),
            movement_phase: self.movement_state.clone(),
        })
    }

    fn analyze_pushup(&mut self, features: &[f32]) -> Result<PoseAnalysis, String> {
        let vertical_accel = features.get(1).unwrap_or(&0.0); // Y轴加速度
        let elbow_flex_estimate = self.estimate_elbow_flexion(features);
        
        self.smoothing_window.push(*vertical_accel);
        if self.smoothing_window.len() > 8 {
            self.smoothing_window.remove(0);
        }
        
        let smoothed_accel: f32 = self.smoothing_window.iter().sum::<f32>() / self.smoothing_window.len() as f32;
        let (new_phase, rep_detected) = self.detect_pushup_phase(smoothed_accel);
        self.movement_state = new_phase;
        
        let form_score = self.calculate_pushup_form_score(elbow_flex_estimate, features);
        
        Ok(PoseAnalysis {
            form_score,
            knee_angle: None,
            elbow_angle: Some(elbow_flex_estimate),
            back_angle: self.estimate_back_angle(features),
            detected_repetition: Some(rep_detected),
            movement_phase: self.movement_state.clone(),
        })
    }

    fn analyze_plank(&mut self, features: &[f32]) -> Result<PoseAnalysis, String> {
        let stability_score = self.calculate_stability_score(features);
        let back_angle = self.estimate_back_angle(features);
        
        Ok(PoseAnalysis {
            form_score: stability_score,
            knee_angle: None,
            elbow_angle: None,
            back_angle,
            detected_repetition: Some(false), // 平板支撑不计算重复次数
            movement_phase: MovementPhase::Rest,
        })
    }

    fn analyze_generic(&mut self, features: &[f32]) -> Result<PoseAnalysis, String> {
        let general_score = features.iter().map(|x| x.abs()).sum::<f32>() / features.len() as f32;
        
        Ok(PoseAnalysis {
            form_score: (general_score * 100.0).min(100.0),
            knee_angle: None,
            elbow_angle: None,
            back_angle: None,
            detected_repetition: Some(false),
            movement_phase: MovementPhase::Rest,
        })
    }

    fn detect_squat_phase(&self, accel: f32) -> (MovementPhase, bool) {
        let mut rep_detected = false;
        
        let new_phase = match self.movement_state {
            MovementPhase::Rest => {
                if accel < -0.5 { MovementPhase::Eccentric } else { MovementPhase::Rest }
            },
            MovementPhase::Eccentric => {
                if accel.abs() < 0.1 { MovementPhase::Bottom } else { MovementPhase::Eccentric }
            },
            MovementPhase::Bottom => {
                if accel > 0.5 { MovementPhase::Concentric } else { MovementPhase::Bottom }
            },
            MovementPhase::Concentric => {
                if accel.abs() < 0.1 {
                    rep_detected = true;
                    MovementPhase::Rest
                } else {
                    MovementPhase::Concentric
                }
            },
        };
        
        (new_phase, rep_detected)
    }

    fn detect_pushup_phase(&self, accel: f32) -> (MovementPhase, bool) {
        let mut rep_detected = false;
        
        let new_phase = match self.movement_state {
            MovementPhase::Rest => {
                if accel > 0.3 { MovementPhase::Eccentric } else { MovementPhase::Rest }
            },
            MovementPhase::Eccentric => {
                if accel.abs() < 0.1 { MovementPhase::Bottom } else { MovementPhase::Eccentric }
            },
            MovementPhase::Bottom => {
                if accel < -0.3 { MovementPhase::Concentric } else { MovementPhase::Bottom }
            },
            MovementPhase::Concentric => {
                if accel.abs() < 0.1 {
                    rep_detected = true;
                    MovementPhase::Rest
                } else {
                    MovementPhase::Concentric
                }
            },
        };
        
        (new_phase, rep_detected)
    }

    fn estimate_knee_flexion(&self, features: &[f32]) -> f32 {
        // 基于加速度数据估算膝关节弯曲角度
        let z_accel = features.get(2).unwrap_or(&0.0);
        let flexion = 90.0 + (z_accel * 30.0); // 简化计算
        flexion.max(30.0).min(150.0)
    }

    fn estimate_elbow_flexion(&self, features: &[f32]) -> f32 {
        let y_accel = features.get(1).unwrap_or(&0.0);
        let flexion = 90.0 - (y_accel * 45.0);
        flexion.max(0.0).min(180.0)
    }

    fn estimate_back_angle(&self, features: &[f32]) -> Option<f32> {
        let x_accel = features.get(0).unwrap_or(&0.0);
        Some((x_accel * 20.0).abs().min(45.0))
    }

    fn calculate_squat_form_score(&self, knee_angle: f32, features: &[f32]) -> f32 {
        let mut score = 100.0;
        
        // 膝关节角度评分
        let ideal_knee_range = 70.0..=90.0;
        if !ideal_knee_range.contains(&knee_angle) {
            score -= (knee_angle - 80.0).abs() * 0.5;
        }
        
        // 稳定性评分
        let stability = self.calculate_stability_score(features);
        score = score * (stability / 100.0);
        
        score.max(0.0).min(100.0)
    }

    fn calculate_pushup_form_score(&self, elbow_angle: f32, features: &[f32]) -> f32 {
        let mut score = 100.0;
        
        // 肘关节角度评分
        if elbow_angle > 120.0 {
            score -= (elbow_angle - 90.0) * 0.3;
        }
        
        // 身体直线度评分
        if let Some(back_angle) = self.estimate_back_angle(features) {
            score -= back_angle * 2.0;
        }
        
        score.max(0.0).min(100.0)
    }

    fn calculate_stability_score(&self, features: &[f32]) -> f32 {
        let variance = features.iter()
            .map(|x| (x - features.iter().sum::<f32>() / features.len() as f32).powi(2))
            .sum::<f32>() / features.len() as f32;
        
        (100.0 - variance * 50.0).max(0.0).min(100.0)
    }
}