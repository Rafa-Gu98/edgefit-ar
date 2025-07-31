# 🚀 FastAPI + Rust High-Performance Edge Computing

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.68+-orange?logo=rust)](https://rust-lang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-009688?logo=fastapi)](https://fastapi.tiangolo.com)

**EdgeFit-AR 🥽 Intelligent Glasses Sports Assistance System**, achieving millisecond-level feedback for sports injury prevention on smart glasses.

## ✨ Project Features

- ⚡ **5ms Level Real-Time Response** - High-performance pose analysis algorithm implemented in Rust
- 🧠 **AI Intelligent Error Correction** - PyTorch LSTM model for real-time detection of motion errors
- 👓 **AR Visualization** - Support for Web/Unity dual interfaces
- 🔋 **Edge Computing** - Local inference, protecting user privacy
- 📊 **Data Source** - Supports UCI HAR fitness data (expandable in the future)

## 🚀 Quick Start

### Environment Requirements

- Python 3.11
- Rust 1.68+ 
- 8GB+ RAM

## 📁 Project Structure

```
EdgeFit-AR/
├── 📊 data_engine/                # Data Processing Core
│   ├── datasets/                   # Dataset Storage
│   │   ├── raw/                    # Raw Data (UCI HAR)
│   │   └── processed/              # Processed Data
│   ├── preprocessing.py            # Data Preprocessing
│   └── setup_datasets.py           # Dataset Configuration
│
├── ⚡ edge_gateway/               # Edge Computing Gateway
│   ├── api/                        # FastAPI
│   │   ├── main.py                 # Service Entry
│   │   ├── model_manager.py        # AI Model Management
│   │   ├── connection_manager.py   # Conncetion
│   │   └── data_adapter.py         # Data Format Conversion
│   └── rust_engine/                # Rust High-Performance Engine 
│       ├── Cargo.toml              # Rust project configuration
│       ├── pyproject.toml          # Python build configuration
│       ├── build.py                # Maturin build
│       └── src/
│            ├── lib.rs             # Rust core implementation            
│            ├── features.rs        # Feature management code
│            ├── health.py          # Health check functionality
│            ├── pose.py            # Pose detection logic
│            └── inference.py       # Inference processing code     
│
├── 🥽 ar_interface/               # AR User Interface
│   └── web_simulator.py            # Web AR Simulator
│
├── 🤖 training/                   # AI Model Training
│   └── train_model.py              # Training Main Script

├── ⚙️ config/                     # Configuration Files (Auto-Generated)
└── 📋 requirements.txt            # Python Dependencies
```

## 🎯 Core Functions

### 1. Real-Time Pose Analysis
- LSTM neural network-based action recognition
- Supports squats, lunges, and other exercise types
- Millisecond-level error detection and feedback

### 2. Multi-Modal Data Fusion
- 6-axis IMU sensor data (accelerometer + gyroscope)
- Real-time data preprocessing and feature extraction
- Adaptive noise filtering

### 3. AR Visualization Interface
- Web version: Ready to use, no additional software needed
- Unity version: Full 3D AR experience (planned)
- Real-time pose correction guidance

## 🛠️ Development Roadmap

- [x] **v0.1** - Basic data processing and AI training
- [x] **v0.2** - Web AR interface and real-time inference
- [x] **v0.3** - Rust performance optimization engine
- [ ] **v0.4** - Unity 3D AR interface
- [ ] **v0.5** - Mobile deployment support
- [ ] **v0.6** - Cloud synchronization and multi-user support

## 🤝 Contribution Guidelines

Welcome to contribute code! Please follow these steps:

1. Fork this project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## 📄 Dataset Support

**UCI HAR Dataset** - Placed in `data_engine/datasets/raw/uci_har/`

## 📞 Contact Information

- **Author**: [Rafa-Gu98](https://github.com/Rafa-Gu98)
- **Email**: rafagr98.dev@gmail.com
- **Project Homepage**: https://github.com/Rafa-Gu98/edgeFit-ar

## 📜 Open Source License

This project uses the [MIT](LICENSE) open source license.

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) - HAR Dataset
- [PyTorch](https://pytorch.org) - ML
- [FastAPI](https://fastapi.tiangolo.com) - Web 
- [Rust](https://rust-lang.org) - Core+

---
⭐ If this project helps you, please give it a Star!