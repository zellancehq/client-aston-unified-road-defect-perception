# 🛣️ Unified Road Defect Perception Platform

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/YOLOv12-8.2.0+-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A comprehensive AI-powered platform for automated road infrastructure analysis and damage detection**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

The **Unified Road Defect Perception Platform** is an advanced deep learning solution that combines state-of-the-art computer vision models for automated road infrastructure monitoring and damage assessment. Built with dual AI approaches, the platform provides both precise damage localization (YOLOv12) and comprehensive damage classification (ResNet50), enabling efficient road maintenance planning and infrastructure audits.

### Why This Project?

- **🚨 Public Safety**: Early detection of road defects prevents accidents and saves lives
- **💰 Cost Efficiency**: Automates labor-intensive manual inspections
- **⚡ Real-time Processing**: Instant damage assessment from images, videos, or live webcam feeds
- **🎯 Precise Localization**: Pinpoints exact damage locations with bounding boxes
- **📊 Comprehensive Analysis**: Multi-label classification for complex damage patterns

---

## ✨ Features

### 🤖 Dual AI-Powered Detection

#### YOLOv12 Object Detection
- **Precise Location Identification**: Detects exact positions of road damage with bounding boxes
- **Real-time Processing**: Fast inference (30+ FPS) for immediate assessment
- **Multi-defect Support**: Simultaneously detects multiple damage types in one image
- **Confidence Scoring**: Provides reliability metrics for each detection
- **Performance**: 84.2% mAP@50, 87.4% precision, 75.6% recall on test set

#### ResNet50 Image Classification
- **Holistic Analysis**: Classifies overall damage patterns across entire surfaces
- **Multi-label Prediction**: Identifies multiple damage types simultaneously
- **Transfer Learning**: Pre-trained on ImageNet with fine-tuning for road defects
- **High Accuracy**: Average accuracy of 85.3% across all defect classes
- **Probability Distribution**: Shows confidence scores for all damage categories

### 🎯 Detected Damage Types

| Damage Type | Description | Severity Impact |
|-------------|-------------|-----------------|
| **Alligator Cracking** | Interconnected cracks resembling alligator skin | High - Structural failure |
| **Linear Cracking** | Longitudinal/transverse fractures in pavement | Medium - Water infiltration risk |
| **Pothole** | Bowl-shaped depressions or cavities | Critical - Vehicle damage hazard |
| **Patching** | Previously repaired areas with different materials | Low - Indicates past issues |
| **Rutting** | Longitudinal depressions in wheel paths | Medium - Drainage problems |

### 🔧 Analysis Modes

1. **📸 Image Detection (YOLO)**: Upload images for precise defect localization
2. **🧠 Image Classification (ResNet)**: Upload images for comprehensive damage assessment
3. **📹 Real-time Webcam Detection**: Live road monitoring through webcam feed
4. **🎥 Video Processing**: Batch analysis of recorded road inspection videos

### 💡 Key Capabilities

- ⚡ **Automated Model Downloads**: Models download automatically on first use
- 🔄 **Intelligent Caching**: No repeated downloads, faster subsequent loading
- 💾 **Offline Support**: Models stored locally for offline usage
- 📱 **Responsive Design**: Works seamlessly across desktop and mobile devices
- 📊 **Rich Visualizations**: Interactive charts, metrics, and result displays
- 📥 **Export Functionality**: Download annotated images and detailed reports
- 🎨 **Professional UI/UX**: Clean, intuitive interface built with Streamlit
- 🛡️ **Error Handling**: Graceful fallbacks and user-friendly error messages

---

## 🚀 Demo

### Detection Examples

**YOLOv12 Object Detection**
- Detects multiple defects with bounding boxes
- Shows confidence scores for each detection
- Provides spatial coordinates for maintenance crews

**ResNet50 Classification**
- Multi-label predictions with probability scores
- Holistic damage assessment
- Visual confidence distribution charts

---

## 📦 Installation

### Prerequisites

- Python 3.10.13
- CUDA-compatible GPU (optional, for faster inference)
- Webcam (optional, for real-time detection)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/zellancehq/client-aston-unified-road-defect-perception.git
cd client-aston-unified-road-defect-perception
```

2. **Create virtual environment** (recommended)
```bash
python -m venv rdd_env
# Windows
rdd_env\Scripts\activate
# Linux/Mac
source rdd_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install system packages** (Linux only)
```bash
# For Ubuntu/Debian
xargs -a packages.txt sudo apt-get install -y
```

5. **Verify installation**
```bash
python -c "import streamlit; import torch; import ultralytics; print('Installation successful!')"
```

---

## 🎮 Usage

### Starting the Application

```bash
streamlit run Home.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Platform

#### 1. Image Detection (YOLO)
```
1. Navigate to "Road damage detection - YOLO" from the sidebar
2. Upload a road image (PNG, JPG, JPEG)
3. Adjust confidence threshold (default: 0.25)
4. Click "Detect Defects"
5. View annotated results and download output
```

#### 2. Image Classification (ResNet)
```
1. Navigate to "Road damage classification - ResNet"
2. Upload a road image
3. Set confidence threshold
4. Click "Classify"
5. Review probability distribution and predictions
```

#### 3. Real-time Detection
```
1. Navigate to "Realtime Detection - YOLO"
2. Grant webcam permissions
3. Adjust detection parameters
4. Start video stream
5. View live damage detection overlay
```

#### 4. Video Processing
```
1. Navigate to "Video Detection - YOLO"
2. Upload a video file
3. Configure detection settings
4. Process video frame by frame
5. Export annotated video
```

---

## 🧠 Models

### YOLOv12 Medium

**Architecture Specifications**
- **Parameters**: 20.1M
- **Input Size**: 640×640 pixels
- **GFLOPs**: 67.1
- **Layers**: 169 fused layers

**Training Details**
- **Dataset**: Road Damage Dataset 2022
- **Images**: 4,844 total (3,386 train / 967 validation / 491 test)
- **Epochs**: 300
- **Batch Size**: 16
- **Optimizer**: AdamW

**Performance Metrics**
| Metric | Validation | Test |
|--------|-----------|------|
| mAP@50 | 84.2% | 82.5% |
| mAP@50-95 | 68.2% | 67.0% |
| Precision | 85.2% | 87.4% |
| Recall | 78.1% | 75.6% |

### ResNet50

**Architecture Specifications**
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Input Size**: 224×224 pixels
- **Parameters**: ~25M
- **Transfer Learning**: Frozen convolutional layers, fine-tuned FC layers

**Training Details**
- **Dataset**: Custom road defect dataset
- **Classes**: 5 (multi-label classification)
- **Epochs**: Fine-tuned for optimal performance
- **Optimizer**: Adam with learning rate scheduling

**Performance Metrics**
| Class | Accuracy |
|-------|----------|
| Depression/Rutting | 98.6% |
| Raveling | 86.9% |
| Pothole | 85.3% |
| Isolated Crack | 80.5% |
| Patch | 75.1% |
| **Average** | **85.3%** |

### Model Downloads

Models are automatically downloaded from GitHub on first use:
- **YOLOv12**: `models/YOLOv12_Road_Defects_Model.pt`
- **ResNet50**: `models/ResNet50_Road_Defects_Model.pth`

Manual download:
```bash
# YOLOv12
wget https://github.com/zellancehq/client-aston-unified-road-defect-perception/releases/download/v1.0/YOLOv12_Road_Defects_Model.pt -P models/

# ResNet50
wget https://github.com/zellancehq/client-aston-unified-road-defect-perception/releases/download/v1.0/ResNet50_Road_Defects_Model.pth -P models/
```

---

## 📊 Performance Analysis

### YOLOv12 Strengths
- ✅ Excellent precision (87.4%) - minimal false positives
- ✅ Real-time processing capability
- ✅ Accurate bounding box localization
- ✅ Robust to varying lighting conditions

### ResNet50 Strengths
- ✅ Outstanding performance on rutting/depression (98.6%)
- ✅ Strong multi-label classification
- ✅ Holistic damage pattern recognition
- ✅ Efficient transfer learning approach

### Use Case Recommendations

**Use YOLOv12 when:**
- Precise damage localization is required
- Real-time processing is needed
- Multiple defects must be detected simultaneously
- Generating maintenance work orders with coordinates

**Use ResNet50 when:**
- Overall road condition assessment is needed
- Multiple damage types coexist
- Prioritizing road segments for inspection
- Statistical analysis of road network health

---

## 🏗️ Project Structure

```
unified-road-defect-perception/
├── Home.py                          # Main Streamlit application entry point
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python version specification
├── packages.txt                     # System-level dependencies
├── models/                          # Pre-trained model files
│   ├── YOLOv12_Road_Defects_Model.pt
│   └── ResNet50_Road_Defects_Model.pth
├── pages/                           # Streamlit multi-page components
│   ├── 1_Road damage detection - YOLO.py
│   ├── 2_Road damage classification - ResNet.py
│   ├── 3_Realtime Detection - YOLO.py
│   └── 4_Video Detection - YOLO.py
├── notebook/                        # Jupyter notebooks for training
│   ├── YOLOv12_Road_Defects_Training.ipynb
│   └── ResNet50_Road-Defects-Training.ipynb
├── reports/                         # Model performance reports
│   ├── yolo/
│   │   ├── YOLOv12_Comprehensive_Report.md
│   │   └── performance_summary.csv
│   └── resnet-50/
│       ├── ResNet50_Comprehensive_Report.md
│       ├── class_accuracies.csv
│       └── epoch_details.csv
├── sample_utils/                    # Utility functions
│   ├── download.py                  # Model download helper
│   └── get_STUNServer.py           # WebRTC server configuration
└── README.md                        # This file
```

---

## 🔬 Technical Details

### Dependencies

**Core Frameworks**
- `torch==2.0.0` - Deep learning framework
- `ultralytics>=8.2.0` - YOLOv8/v12 implementation
- `streamlit==1.28.0` - Web application framework

**Computer Vision**
- `opencv-python-headless==4.8.0.74` - Image processing
- `Pillow==9.5.0` - Image handling

**Real-time Processing**
- `streamlit-webrtc>=0.47.1` - WebRTC integration
- `av>=11.0.0` - Video processing

**Utilities**
- `numpy==1.24.3` - Numerical operations
- `requests==2.28.2` - HTTP requests for model downloads

### Hardware Requirements

**Minimum**
- CPU: Dual-core processor
- RAM: 8GB
- Storage: 2GB free space
- GPU: Optional (CPU inference supported)

**Recommended**
- CPU: Quad-core processor (Intel i5/AMD Ryzen 5 or better)
- RAM: 16GB
- Storage: 5GB free space
- GPU: NVIDIA GPU with 4GB+ VRAM (for real-time detection)

---

## 📚 Documentation

### Comprehensive Reports

Detailed technical documentation available in the `reports/` directory:

- **[YOLOv12 Comprehensive Report](reports/yolo/YOLOv12_Comprehensive_Report.md)**
  - Architecture overview
  - Training methodology
  - Performance metrics
  - Ablation studies

- **[ResNet50 Comprehensive Report](reports/resnet-50/ResNet50_Comprehensive_Report.md)**
  - Transfer learning approach
  - Fine-tuning strategy
  - Class-wise performance analysis
  - Confusion matrix insights

### Training Notebooks

Jupyter notebooks for model training and experimentation:
- `notebook/YOLOv12_Road_Defects_Training.ipynb` - YOLOv12 training pipeline
- `notebook/ResNet50_Road-Defects-Training.ipynb` - ResNet50 training workflow

---

## 💼 Use Cases

### Municipal Road Maintenance
- **Priority Scheduling**: Rank road segments by damage severity
- **Resource Allocation**: Optimize maintenance crew deployment
- **Budget Planning**: Estimate repair costs based on damage extent

### Infrastructure Audits
- **Periodic Assessments**: Automated quarterly road condition surveys
- **Compliance Reporting**: Generate standardized damage reports
- **Historical Tracking**: Monitor road deterioration over time

### Construction Quality Control
- **Post-Construction Verification**: Ensure newly paved roads meet standards
- **Warranty Claims**: Document defects within warranty periods
- **Contractor Evaluation**: Assess construction quality objectively

### Insurance & Liability
- **Damage Documentation**: Create timestamped evidence for claims
- **Risk Assessment**: Identify high-risk road sections
- **Legal Evidence**: Provide objective damage records for disputes

### Research & Development
- **Pavement Studies**: Analyze material performance and longevity
- **Climate Impact**: Study weather effects on road deterioration
- **Algorithm Development**: Benchmark new detection algorithms

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Model Download Fails**
```bash
# Manual download
cd models/
wget https://github.com/zellancehq/client-aston-unified-road-defect-perception/releases/download/v1.0/YOLOv12_Road_Defects_Model.pt
```

**2. CUDA Out of Memory**
```python
# Reduce batch size or use CPU
# Set in Streamlit sidebar: Device = "CPU"
```

**3. Webcam Not Detected**
```bash
# Check permissions
# Windows: Settings > Privacy > Camera
# Linux: ls /dev/video*
```

**4. Streamlit Port Already in Use**
```bash
# Use different port
streamlit run Home.py --server.port 8502
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **YOLOv8/v12**: [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) by Ultralytics
- **Streamlit**: [Apache 2.0](https://github.com/streamlit/streamlit/blob/develop/LICENSE) by Streamlit Inc.
- **PyTorch**: [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- **ResNet**: PyTorch implementation with ImageNet pre-trained weights

---

## 📞 Contact & Support

**Author**: Tasmia Azim  
**Email**: tasmiayemna@gmail.com  
**GitHub**: [@zellancehq](https://github.com/zellancehq)  
**Project Repository**: [client-aston-unified-road-defect-perception](https://github.com/zellancehq/client-aston-unified-road-defect-perception)

### Getting Help

- 📖 Check the [Documentation](#-documentation) section
- 🐛 Report bugs via [GitHub Issues](https://github.com/zellancehq/client-aston-unified-road-defect-perception/issues)
- 💬 Ask questions in [Discussions](https://github.com/zellancehq/client-aston-unified-road-defect-perception/discussions)
- 📧 Email for collaboration inquiries

---

## 🙏 Acknowledgments

- **Ultralytics** for the outstanding YOLOv8/v12 framework
- **Streamlit** for the intuitive web application framework
- **PyTorch** team for the powerful deep learning library
- **Road Damage Dataset** contributors for the training data
- **Microsoft Research** for ResNet architecture
- Open source community for continuous support and feedback

---

## 📈 Future Roadmap

- [ ] **Mobile Application**: Native Android/iOS apps
- [ ] **Edge Deployment**: Raspberry Pi and edge device support
- [ ] **3D Road Mapping**: Integration with GPS for spatial mapping
- [ ] **Severity Estimation**: Automated damage severity scoring
- [ ] **Multi-language Support**: Interface localization
- [ ] **Cloud API**: RESTful API for integration
- [ ] **Database Integration**: Store and track historical data
- [ ] **Report Generation**: Automated PDF/Excel reports
- [ ] **Real-time Alerts**: Notification system for critical damage
- [ ] **Model Optimization**: Quantization and pruning for faster inference

---

## 📊 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{unified_road_defect_perception,
  author = {Azim, Tasmia},
  title = {Unified Road Defect Perception Platform},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/zellancehq/client-aston-unified-road-defect-perception}
}
```

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for safer roads worldwide

[Back to Top](#-unified-road-defect-perception-platform)

</div>
