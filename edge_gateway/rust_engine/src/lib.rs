use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

pub mod inference;
pub mod pose;
pub mod health;
pub mod features;

use inference::InferenceEngine;
use pose::PoseAnalyzer;
use health::HealthMonitor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorData {
    pub timestamp: u64,
    pub accelerometer: [f32; 3],
    pub gyroscope: [f32; 3],
    pub magnetometer: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseKeypoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub exercise_type: String,
    pub repetitions: u32,
    pub form_score: f32,
    pub errors: Vec<FormError>,
    pub calories_burned: f32,
    pub muscle_activation: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormError {
    pub error_type: String,
    pub severity: String,
    pub timestamp: u64,
    pub suggestion: String,
}

#[pyclass]
pub struct EdgeFitEngine {
    inference_engine: InferenceEngine,
    pose_analyzer: PoseAnalyzer,
    health_monitor: HealthMonitor,
    current_exercise: Option<String>,
    rep_count: u32,
    session_start: Option<u64>,
}

#[pymethods]
impl EdgeFitEngine {
    #[new]
    pub fn new(model_path: &str) -> PyResult<Self> {
        let inference_engine = InferenceEngine::new(model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;
        
        Ok(EdgeFitEngine {
            inference_engine,
            pose_analyzer: PoseAnalyzer::new(),
            health_monitor: HealthMonitor::new(),
            current_exercise: None,
            rep_count: 0,
            session_start: None,
        })
    }

    pub fn process_sensor_data(&mut self, py: Python, data: &Bound<PyList>) -> PyResult<PyObject> {
        // 转换Python数据到Rust结构
        let mut sensor_data = Vec::new();
        
        for item in data.iter() {
            let dict = item.downcast::<PyDict>()?;
            
            let sensor = SensorData {
                timestamp: dict.get_item("timestamp")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'timestamp'"))?
                    .extract()?,
                accelerometer: dict.get_item("accelerometer")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'accelerometer'"))?
                    .extract()?,
                gyroscope: dict.get_item("gyroscope")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'gyroscope'"))?
                    .extract()?,
                magnetometer: dict.get_item("magnetometer")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'magnetometer'"))?
                    .extract()?,
            };
            sensor_data.push(sensor);
        }

        // 执行推理
        let analysis = self.analyze_movement(&sensor_data)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        // 转换结果回Python对象
        self.analysis_result_to_pyobject(py, &analysis)
    }

    pub fn start_exercise_session(&mut self, exercise_type: String) -> PyResult<()> {
        self.current_exercise = Some(exercise_type);
        self.rep_count = 0;
        self.session_start = Some(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("System time error: {}", e)))?
            .as_millis() as u64);
        Ok(())
    }

    pub fn stop_exercise_session(&mut self) -> PyResult<HashMap<String, f32>> {
        let mut session_summary = HashMap::new();
        
        if let Some(start_time) = self.session_start {
            let duration = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("System time error: {}", e)))?
                .as_millis() as u64 - start_time;
            
            session_summary.insert("duration_ms".to_string(), duration as f32);
            session_summary.insert("total_reps".to_string(), self.rep_count as f32);
            
            // 估算卡路里消耗
            let calories = self.estimate_calories_burned(duration, self.rep_count);
            session_summary.insert("calories_burned".to_string(), calories);
        }

        self.current_exercise = None;
        self.rep_count = 0;
        self.session_start = None;

        Ok(session_summary)
    }

    pub fn get_health_metrics(&self) -> PyResult<HashMap<String, f32>> {
        Ok(self.health_monitor.get_current_metrics())
    }

    pub fn get_fitness_level(&self) -> PyResult<String> {
        Ok(self.health_monitor.get_fitness_level())
    }

    pub fn reset_session_metrics(&mut self) -> PyResult<()> {
        self.health_monitor.reset_session_metrics();
        Ok(())
    }

    pub fn get_model_info(&self) -> PyResult<HashMap<String, String>> {
        Ok(self.inference_engine.get_model_info())
    }

    pub fn evaluate_model(&self, py: Python, test_data: &Bound<PyList>, test_labels: &Bound<PyList>) -> PyResult<HashMap<String, f32>> {
        // 转换测试数据
        let mut features_batch = Vec::new();
        for item in test_data.iter() {
            let features: Vec<f32> = item.extract()?;
            features_batch.push(features);
        }

        let mut labels = Vec::new();
        for item in test_labels.iter() {
            let label: String = item.extract()?;
            labels.push(label);
        }

        // 执行评估
        let metrics = self.inference_engine.evaluate_model(&features_batch, &labels)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(metrics)
    }
}

impl EdgeFitEngine {
    // 手动转换AnalysisResult到Python对象
    fn analysis_result_to_pyobject(&self, py: Python, analysis: &AnalysisResult) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("exercise_type", &analysis.exercise_type)?;
        dict.set_item("repetitions", analysis.repetitions)?;
        dict.set_item("form_score", analysis.form_score)?;
        dict.set_item("calories_burned", analysis.calories_burned)?;
        
        // 转换errors
        let errors_list = PyList::empty_bound(py);
        for error in &analysis.errors {
            let error_dict = PyDict::new_bound(py);
            error_dict.set_item("error_type", &error.error_type)?;
            error_dict.set_item("severity", &error.severity)?;
            error_dict.set_item("timestamp", error.timestamp)?;
            error_dict.set_item("suggestion", &error.suggestion)?;
            errors_list.append(error_dict)?;
        }
        dict.set_item("errors", errors_list)?;
        
        // 转换muscle_activation
        let muscle_dict = PyDict::new_bound(py);
        for (muscle, activation) in &analysis.muscle_activation {
            muscle_dict.set_item(muscle, *activation)?;
        }
        dict.set_item("muscle_activation", muscle_dict)?;
        
        Ok(dict.to_object(py))
    }

    fn analyze_movement(&mut self, sensor_data: &[SensorData]) -> Result<AnalysisResult, String> {
        if self.current_exercise.is_none() {
            return Err("No exercise session started".to_string());
        }
        let exercise_type = self.current_exercise.as_ref().unwrap().clone();

        // 特征提取
        let features = features::extract_features(sensor_data)?;

        // 运动分类
        let _inferred_type = self.inference_engine.classify_exercise(&features)?;
        // 注：可以在这里比较inferred_type和exercise_type的一致性

        // 姿态分析
        let pose_analysis = self.pose_analyzer.analyze_pose(&features, &exercise_type)?;

        // 动作计数
        if let Some(new_rep) = pose_analysis.detected_repetition {
            if new_rep {
                self.rep_count += 1;
            }
        }

        // 健康监测
        self.health_monitor.update_metrics(sensor_data);

        // 错误检测
        let current_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| format!("System time error: {}", e))?
            .as_millis() as u64;
        let errors = self.detect_form_errors(&pose_analysis, &exercise_type, current_timestamp)?;

        // 肌肉激活度估算
        let muscle_activation = self.estimate_muscle_activation(&exercise_type, &pose_analysis);

        Ok(AnalysisResult {
            exercise_type,
            repetitions: self.rep_count,
            form_score: pose_analysis.form_score,
            errors,
            calories_burned: self.estimate_calories_burned(
                sensor_data.len() as u64 * 20, // 假设20ms间隔
                self.rep_count
            ),
            muscle_activation,
        })
    }

    fn detect_form_errors(&self, pose_analysis: &pose::PoseAnalysis, exercise_type: &str, timestamp: u64) -> Result<Vec<FormError>, String> {
        let mut errors = Vec::new();

        match exercise_type {
            "squat" => {
                // 深蹲错误检测
                if pose_analysis.knee_angle.unwrap_or(90.0) < 70.0 {
                    errors.push(FormError {
                        error_type: "knee_valgus".to_string(),
                        severity: "moderate".to_string(),
                        timestamp,
                        suggestion: "保持膝盖与脚尖方向一致，避免内扣".to_string(),
                    });
                }

                if pose_analysis.back_angle.unwrap_or(0.0) > 30.0 {
                    errors.push(FormError {
                        error_type: "forward_lean".to_string(),
                        severity: "high".to_string(),
                        timestamp,
                        suggestion: "保持躯干挺直，避免过度前倾".to_string(),
                    });
                }
            },
            "pushup" => {
                // 俯卧撑错误检测
                if pose_analysis.elbow_angle.unwrap_or(90.0) > 120.0 {
                    errors.push(FormError {
                        error_type: "incomplete_range".to_string(),
                        severity: "low".to_string(),
                        timestamp,
                        suggestion: "增加下降深度，获得更好的训练效果".to_string(),
                    });
                }
            },
            _ => {}
        }

        Ok(errors)
    }

    fn estimate_muscle_activation(&self, exercise_type: &str, pose_analysis: &pose::PoseAnalysis) -> HashMap<String, f32> {
        let mut activation = HashMap::new();

        match exercise_type {
            "squat" => {
                // 基于关节角度估算肌肉激活度
                let knee_activation = 1.0 - (pose_analysis.knee_angle.unwrap_or(90.0) - 70.0).abs() / 50.0;
                activation.insert("quadriceps".to_string(), knee_activation.max(0.0).min(1.0));
                activation.insert("glutes".to_string(), 0.85);
                activation.insert("hamstrings".to_string(), 0.62);
            },
            "pushup" => {
                activation.insert("pectorals".to_string(), 0.80);
                activation.insert("triceps".to_string(), 0.75);
                activation.insert("deltoids".to_string(), 0.65);
            },
            _ => {
                activation.insert("general".to_string(), 0.5);
            }
        }

        activation
    }

    fn estimate_calories_burned(&self, duration_ms: u64, reps: u32) -> f32 {
        // 简化的卡路里计算（实际应该基于用户体重、年龄等）
        let base_rate = 0.05; // 每毫秒基础消耗
        let rep_bonus = reps as f32 * 0.8; // 每个重复动作额外消耗
        
        (duration_ms as f32 * base_rate + rep_bonus).max(0.0)
    }
}

#[pymodule]
fn rust_engine(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<EdgeFitEngine>()?;
    Ok(())
}