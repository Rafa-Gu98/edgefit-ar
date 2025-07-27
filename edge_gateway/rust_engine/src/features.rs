use crate::SensorData;
use std::f32::consts::PI;

pub fn extract_features(sensor_data: &[SensorData]) -> Result<Vec<f32>, String> {
    if sensor_data.is_empty() {
        return Err("Empty sensor data".to_string());
    }

    let mut features = Vec::new();
    
    // 时域特征
    features.extend(extract_time_domain_features(sensor_data));
    
    // 频域特征
    features.extend(extract_frequency_domain_features(sensor_data));
    
    // 统计特征
    features.extend(extract_statistical_features(sensor_data));
    
    // 姿态特征
    features.extend(extract_orientation_features(sensor_data));
    
    Ok(features)
}

fn extract_time_domain_features(data: &[SensorData]) -> Vec<f32> {
    let mut features = Vec::new();
    
    // 加速度幅值
    for axis in 0..3 {
        let values: Vec<f32> = data.iter().map(|d| d.accelerometer[axis]).collect();
        features.push(calculate_mean(&values));
        features.push(calculate_std(&values));
        features.push(calculate_rms(&values));
    }
    
    // 陀螺仪幅值
    for axis in 0..3 {
        let values: Vec<f32> = data.iter().map(|d| d.gyroscope[axis]).collect();
        features.push(calculate_mean(&values));
        features.push(calculate_std(&values));
    }
    
    features
}

fn extract_frequency_domain_features(data: &[SensorData]) -> Vec<f32> {
    let mut features = Vec::new();
    
    // 简化的频域特征（实际应该使用FFT）
    for axis in 0..3 {
        let values: Vec<f32> = data.iter().map(|d| d.accelerometer[axis]).collect();
        let dominant_freq = estimate_dominant_frequency(&values);
        let spectral_energy = calculate_spectral_energy(&values);
        
        features.push(dominant_freq);
        features.push(spectral_energy);
    }
    
    features
}

fn extract_statistical_features(data: &[SensorData]) -> Vec<f32> {
    let mut features = Vec::new();
    
    // 加速度矢量幅值
    let magnitudes: Vec<f32> = data.iter().map(|d| {
        (d.accelerometer[0].powi(2) + d.accelerometer[1].powi(2) + d.accelerometer[2].powi(2)).sqrt()
    }).collect();
    
    features.push(calculate_mean(&magnitudes));
    features.push(calculate_std(&magnitudes));
    features.push(calculate_min(&magnitudes));
    features.push(calculate_max(&magnitudes));
    features.push(calculate_skewness(&magnitudes));
    features.push(calculate_kurtosis(&magnitudes));
    
    features
}

fn extract_orientation_features(data: &[SensorData]) -> Vec<f32> {
    let mut features = Vec::new();
    
    // 重力方向估算
    if !data.is_empty() {
        let gravity = estimate_gravity_direction(data);
        
        // 先计算倾斜角度，然后扩展gravity，以避免移动后借用
        let tilt_angles = calculate_tilt_angles(&gravity);
        features.extend(gravity);
        features.extend(tilt_angles);
    }
    
    features
}

// 辅助函数
fn calculate_mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn calculate_std(values: &[f32]) -> f32 {
    let mean = calculate_mean(values);
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

fn calculate_rms(values: &[f32]) -> f32 {
    (values.iter().map(|x| x.powi(2)).sum::<f32>() / values.len() as f32).sqrt()
}

fn calculate_min(values: &[f32]) -> f32 {
    values.iter().fold(f32::INFINITY, |a, &b| a.min(b))
}

fn calculate_max(values: &[f32]) -> f32 {
    values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
}

fn calculate_skewness(values: &[f32]) -> f32 {
    let mean = calculate_mean(values);
    let std = calculate_std(values);
    
    if std == 0.0 { return 0.0; }
    
    let skew = values.iter()
        .map(|x| ((x - mean) / std).powi(3))
        .sum::<f32>() / values.len() as f32;
    skew
}

fn calculate_kurtosis(values: &[f32]) -> f32 {
    let mean = calculate_mean(values);
    let std = calculate_std(values);
    
    if std == 0.0 { return 0.0; }
    
    let kurt = values.iter()
        .map(|x| ((x - mean) / std).powi(4))
        .sum::<f32>() / values.len() as f32 - 3.0;
    kurt
}

fn estimate_dominant_frequency(values: &[f32]) -> f32 {
    // 简化的主频估算（实际应该使用FFT）
    let mut max_amplitude = 0.0;
    let mut dominant_freq = 0.0;
    
    for freq in 1..10 {
        let amplitude = values.iter()
            .enumerate()
            .map(|(i, &x)| x * (2.0 * PI * freq as f32 * i as f32 / values.len() as f32).cos())
            .sum::<f32>().abs();
        
        if amplitude > max_amplitude {
            max_amplitude = amplitude;
            dominant_freq = freq as f32;
        }
    }
    
    dominant_freq
}

fn calculate_spectral_energy(values: &[f32]) -> f32 {
    values.iter().map(|x| x.powi(2)).sum::<f32>()
}

fn estimate_gravity_direction(data: &[SensorData]) -> Vec<f32> {
    let mut gravity = [0.0f32; 3];
    
    for d in data {
        for i in 0..3 {
            gravity[i] += d.accelerometer[i];
        }
    }
    
    for i in 0..3 {
        gravity[i] /= data.len() as f32;
    }
    
    gravity.to_vec()
}

fn calculate_tilt_angles(gravity: &[f32]) -> Vec<f32> {
    let roll = gravity[1].atan2(gravity[2]);
    let pitch = (-gravity[0]).atan2((gravity[1].powi(2) + gravity[2].powi(2)).sqrt());
    
    vec![roll, pitch]
}