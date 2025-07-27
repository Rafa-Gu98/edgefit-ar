# EdgeFit-AR 🥽 智能眼镜运动辅助系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.68+-orange?logo=rust)](https://rust-lang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-009688?logo=fastapi)](https://fastapi.tiangolo.com)

**边缘计算：融合 Python 生态与 Rust 性能的实时运动姿态分析系统**，在智能眼镜上实现毫秒级反馈的运动损伤预防解决方案。

## ✨ 项目特色

- ⚡ **5ms 级实时响应** - Rust 实现的高性能姿态分析算法
- 🧠 **AI 智能纠错** - PyTorch LSTM 模型实时检测运动错误
- 👓 **AR 可视化** - Web/Unity 双重界面支持
- 🔋 **边缘计算** - 本地推理，保护用户隐私
- 📊 **数据源** - 支持 UCI HAR健身数据(后续可拓展)

## 🚀 快速开始

### 环境要求

- Python 3.11
- Rust 1.68+ 
- 8GB+ RAM


## 📁 项目结构

```
EdgeFit-AR/
├── 📊 data_engine/              # 数据处理核心
│   ├── datasets/                # 数据集存储
│   │   ├── raw/                # 原始数据 (UCI HAR)
│   │   ├── processed/          # 处理后数据
│   │   └── simulated/          # 模拟生成数据
│   ├── preprocessing.py        # 数据预处理
│   ├── simulator.py           # 传感器数据模拟
│   └── setup_datasets.py      # 数据集配置
│
├── ⚡ edge_gateway/             # 边缘计算网关
│   ├── api/                   # FastAPI
│   ├── rust_engine/           # Rust 高性能引擎 (开发中)
│   ├── main.py               # 服务入口
│   ├── model_manager.py      # AI 模型管理
│   └── data_adapter.py       # 数据格式转换
│
├── 🥽 ar_interface/            # AR 用户界面
│   ├── web_simulator.py      # Web AR 模拟器
│   └── __init__.py
│
├── 🤖 training/               # AI 模型训练
│   └── train_model.py        # 训练主脚本
│
├── 🔧 hardware_simulator/     # 硬件模拟器
│   ├── sensor_emulator.py    # 传感器数据模拟
│   ├── ar_simulator.py       # AR 设备模拟
│   └── data_generator.py     # 数据生成工具
│
├── ⚙️ config/                 # 配置文件 (自动生成)
└── 📋 requirements.txt        # Python 依赖
```

## 🎯 核心功能

### 1. 实时姿态分析
- 基于 LSTM 神经网络的动作识别
- 支持深蹲、弓步等多种运动类型
- 毫秒级错误检测和反馈

### 2. 多模态数据融合
- 6 轴 IMU 传感器数据 (加速度计 + 陀螺仪)
- 实时数据预处理和特征提取
- 自适应噪声过滤

### 3. AR 可视化界面
- Web 版本：即开即用，无需额外软件
- Unity 版本：完整 3D AR 体验 (规划中)
- 实时姿态纠错指导


## 🛠️ 开发路线图

- [x] **v0.1** - 基础数据处理和 AI 训练
- [x] **v0.2** - Web AR 界面和实时推理
- [ ] **v0.3** - Rust 性能优化引擎
- [ ] **v0.4** - Unity 3D AR 界面
- [ ] **v0.5** - 移动端部署支持
- [ ] **v0.6** - 云端同步和多用户支持

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request


## 📄 数据集支持

**UCI HAR Dataset** - 放置在 `data_engine/datasets/raw/uci_har/`

## 📞 联系方式

- **作者**: [Rafa-Gu98](https://github.com/Rafa-Gu98)
- **邮箱**: rafagr98.dev@gmail.com
- **项目主页**: https://github.com/Rafa-Gu98/edgeFit-ar

## 📜 开源协议

本项目采用 [MIT](LICENSE) 开源协议。

## 🙏 致谢

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) - HAR 数据集
- [PyTorch](https://pytorch.org) - ML
- [FastAPI](https://fastapi.tiangolo.com) - Web 
- [Rust](https://rust-lang.org) - Core+

---
⭐ 如果这个项目对你有帮助，请给个 Star！