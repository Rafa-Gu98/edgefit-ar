use tch::{Tensor, Device, Kind, nn, nn::Module};
use std::path::Path;
use std::collections::HashMap;

pub struct InferenceEngine {
    exercise_classifier: nn::VarStore,
    pose_estimator: nn::VarStore,
    device: Device,
    model: Box<dyn nn::Module>,
}

impl InferenceEngine {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let device = Device::cuda_if_available();
        
        let mut exercise_classifier = nn::VarStore::new(device);
        let mut pose_estimator = nn::VarStore::new(device);
        
        // 构建模型
        let vs = nn::VarStore::new(device);
        let model = Self::build_model(&vs.root())?;
        
        // 尝试加载预训练模型权重
        if Path::new(&format!("{}/exercise_classifier.pt", model_path)).exists() {
            exercise_classifier.load(&format!("{}/exercise_classifier.pt", model_path))
                .map_err(|e| format!("Failed to load exercise classifier: {}", e))?;
        }
        
        if Path::new(&format!("{}/pose_estimator.pt", model_path)).exists() {
            pose_estimator.load(&format!("{}/pose_estimator.pt", model_path))
                .map_err(|e| format!("Failed to load pose estimator: {}", e))?;
        }
        
        // 如果存在统一模型文件，也尝试加载
        if Path::new(model_path).exists() && model_path.ends_with(".pt") {
            vs.load(model_path)
                .map_err(|e| format!("Failed to load model weights: {}", e))?;
        }
        
        Ok(InferenceEngine {
            exercise_classifier,
            pose_estimator,
            device,
            model: Box::new(model),
        })
    }

    pub fn classify_exercise(&self, features: &[f32]) -> Result<String, String> {
        if features.is_empty() {
            return Err("Empty features".to_string());
        }

        // 使用 from_slice 替代 of_slice，并正确处理错误
        let input = Tensor::from_slice(features)
            .to_device(self.device)
            .unsqueeze(0); // 添加 batch 维度
        
        let output = tch::no_grad(|| {
            self.model.forward(&input)
        });
        
        // 获取预测类别
        let predicted_class = output.argmax(1, false);
        let class_idx: i64 = predicted_class.int64_value(&[0]);
        
        self.class_to_exercise_type(class_idx)
    }

    pub fn predict_form_score(&self, features: &[f32]) -> Result<f32, String> {
        if features.is_empty() {
            return Err("Empty features".to_string());
        }

        let input = Tensor::from_slice(features)
            .to_device(self.device)
            .unsqueeze(0);
        
        let output = tch::no_grad(|| {
            self.model.forward(&input)
        });
        
        // 假设输出是回归分数 (0-1)
        let score: f64 = output.double_value(&[0, 0]);
        Ok(score.max(0.0).min(1.0) as f32)
    }

    pub fn estimate_pose(&self, features: &[f32]) -> Result<Vec<(f32, f32, f32)>, String> {
        if features.is_empty() {
            return Err("Empty features".to_string());
        }

        let input = Tensor::from_slice(features)
            .to_device(self.device)
            .unsqueeze(0);

        let output = tch::no_grad(|| {
            // 调整输出形状以适应姿态估计
            let pose_output = self.model.forward(&input);
            // 假设输出17个关键点，每个3个坐标
            pose_output.view([-1, 17, 3])
        });

        // 将张量转换为Vec<f32>
        let keypoints_flat: Vec<f32> = Vec::from(output.flatten(0, -1));
        let mut result = Vec::new();

        for chunk in keypoints_flat.chunks(3) {
            if chunk.len() == 3 {
                result.push((chunk[0], chunk[1], chunk[2]));
            }
        }

        Ok(result)
    }

    fn build_model(vs: &nn::Path) -> Result<impl nn::Module, String> {
        // 构建神经网络模型
        let input_size = 50; // 根据特征数量调整
        let hidden_size = 128;
        let num_classes = 5; // 支持的运动类型数量
        
        let model = nn::seq()
            .add(nn::linear(vs / "layer1", input_size, hidden_size, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::dropout(vs / "dropout1", 0.3, true))
            .add(nn::linear(vs / "layer2", hidden_size, hidden_size, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::dropout(vs / "dropout2", 0.3, true))
            .add(nn::linear(vs / "layer3", hidden_size, num_classes, Default::default()))
            .add_fn(|xs| xs.softmax(-1, Kind::Float));
        
        Ok(model)
    }

    fn class_to_exercise_type(&self, class_idx: i64) -> Result<String, String> {
        let exercise_types = vec!["squat", "pushup", "plank", "lunge", "jumping_jack"];
        
        if class_idx >= 0 && (class_idx as usize) < exercise_types.len() {
            Ok(exercise_types[class_idx as usize].to_string())
        } else {
            Err(format!("Unknown class index: {}", class_idx))
        }
    }

    // 添加批量推理功能
    pub fn classify_exercises_batch(&self, features_batch: &[Vec<f32>]) -> Result<Vec<String>, String> {
        if features_batch.is_empty() {
            return Err("Empty batch".to_string());
        }

        let mut results = Vec::new();
        
        for features in features_batch {
            match self.classify_exercise(features) {
                Ok(exercise_type) => results.push(exercise_type),
                Err(e) => return Err(format!("Batch processing failed: {}", e)),
            }
        }
        
        Ok(results)
    }

    // 添加模型评估功能
    pub fn evaluate_model(&self, test_features: &[Vec<f32>], test_labels: &[String]) -> Result<HashMap<String, f32>, String> {
        if test_features.len() != test_labels.len() {
            return Err("Features and labels length mismatch".to_string());
        }

        let mut correct = 0;
        let mut total = 0;
        let mut class_correct: HashMap<String, i32> = HashMap::new();
        let mut class_total: HashMap<String, i32> = HashMap::new();

        for (features, true_label) in test_features.iter().zip(test_labels.iter()) {
            match self.classify_exercise(features) {
                Ok(predicted_label) => {
                    total += 1;
                    *class_total.entry(true_label.clone()).or_insert(0) += 1;
                    
                    if predicted_label == *true_label {
                        correct += 1;
                        *class_correct.entry(true_label.clone()).or_insert(0) += 1;
                    }
                },
                Err(_) => continue,
            }
        }

        let mut metrics = HashMap::new();
        metrics.insert("overall_accuracy".to_string(), correct as f32 / total as f32);

        // 计算每个类别的准确率
        for (class, &class_total_count) in &class_total {
            let class_correct_count = class_correct.get(class).unwrap_or(&0);
            let class_accuracy = *class_correct_count as f32 / class_total_count as f32;
            metrics.insert(format!("{}_accuracy", class), class_accuracy);
        }

        Ok(metrics)
    }

    // 获取模型信息
    pub fn get_model_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("device".to_string(), format!("{:?}", self.device));
        info.insert("supported_exercises".to_string(), 
                   "squat,pushup,plank,lunge,jumping_jack".to_string());
        info.insert("input_features".to_string(), "50".to_string());
        info.insert("model_type".to_string(), "neural_network".to_string());
        info
    }
}