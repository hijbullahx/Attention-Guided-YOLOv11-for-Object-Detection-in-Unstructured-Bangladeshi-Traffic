# Real-Time Attention-Guided Object Detection and Dynamic Feature Fusion for Dense Unstructured Traffic in Developing Regions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Paper](https://img.shields.io/badge/Springer%20Nature-JRTIP-brightgreen.svg)](https://link.springer.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20Ready-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)

Official repository for the research paper:  
**"Real-Time Attention-Guided Object Detection and Dynamic Feature Fusion for Dense Unstructured Traffic in Developing Regions"**  
Submitted to: *Journal of Real-Time Image Processing* (JRTIP), **Springer Nature**.

---

## 📌 Abstract & Problem Statement

Automated visual perception is an indispensable cornerstone for **Intelligent Transportation Systems (ITS)** and **Advanced Driver Assistance Systems (ADAS)**. However, conventional state-of-the-art vision models suffer substantial performance degradation when transferred to the highly chaotic and unstructured roadways of South Asian megacities (e.g., Dhaka, Bangladesh). These environments feature severe class heterogeneity, total absence of lane discipline, extreme visual occlusion, and drastic scale disparities—where high-speed motorized vehicles share dense roadway corridors with low-speed non-motorized transports such as human-powered rickshaws, vans, and three-wheeled auto-rickshaws (CNGs).

To resolve these challenges, we present an enhanced and optimized **YOLOv11** framework tailored for edge deployment in dense unstructured traffic environments.

---

## 🚀 Key Highlights & Novel Contributions

1. **Strategic Attention Guidance (CBAM at Layer 23):**  
   We integrate an optimized Convolutional Block Attention Module (CBAM) sequentially coupling channel and spatial attention into the deep feature extraction stage (Layer 23, P5 feature map). This suppresses background visual clutter and sharpens the structural boundaries of occluded vehicles with only a **+0.16% parameter overhead (+32,866 parameters)**.
2. **Dynamic Channel Fusion Module (DCFM):**  
   Designed with parallel multi-scale dilated convolutional branches (dilations 1, 2, 5) and $1\times 1$ linear fusion to adaptively reconcile multi-scale disparities between diminutive bicycles and massive commercial buses.
3. **Comprehensive Regional Benchmark (265,698 Bounding Boxes):**  
   Curated from three regional datasets (*Mendeley Bangladeshi Traffic*, *DhakaAI*, and *Poribohon-BD*), spanning **23,648 images** and **9 localized classes**, partitioned video-wise (80% train / 20% validation) to prevent temporal data leakage.
4. **State-of-the-Art Accuracy:**  
   Attains **75.70% mAP@0.50** (53.20% mAP@0.50:0.95), achieving **90.8% AP@0.50** on non-motorized Rickshaws and a **+1.50% boost** on minority Trucks.
5. **Ultra-Low Edge Latency (212.7 FPS on Tesla T4):**  
   Pure inference latency is reduced to **4.7 ms per frame** (~212.7 FPS on NVIDIA Tesla T4 GPU), outperforming baseline latency due to attention-guided background suppression reducing candidate bounding box queues during Non-Maximum Suppression (NMS).
6. **Deployable Edge Prototype:**  
   Fully containerized, interactive real-time surveillance dashboard built on Streamlit with live vehicular census counters and hardware telemetry.

---

## 🏛️ System Architecture

```text
Input Image (640x640x3)
      │
┌─────▼────────────────────────────────────────────────────────┐
│  YOLOv11 Backbone (P1 - P5 / Layers 0-10)                    │
└─────┬────────────────────────────────────────────────────────┘
      │
┌─────▼────────────────────────────────────────────────────────┐
│  PANet Feature Aggregation Neck (Layers 11-22)               │
└─────┬────────────────────────────────────────────────────────┘
      │
┌─────▼────────────────────────────────────────────────────────┐
│  ★ LAYER 23: CONVOLUTIONAL BLOCK ATTENTION MODULE (CBAM)     │
│   ├── Channel Attention: AvgPool + MaxPool -> Shared MLP      │
│   └── Spatial Attention: 7x7 Conv on Channel Feature Maps    │
└─────┬────────────────────────────────────────────────────────┘
      │
┌─────▼────────────────────────────────────────────────────────┐
│  Decoupled Anchor-Free Detection Head (Layer 24)             │
│   ├── Bounding Box Regression (DFL + CIoU Loss)              │
│   └── 9-Class Classification Output                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Evaluation Results

### Overall Model Performance Comparison (50 Epochs)

| Model Architecture | Params (M) | FLOPs (G) | mAP@0.50 (%) | mAP@0.50:0.95 (%) | Latency (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv8m Baseline | 25.90 | 78.9 | 73.80% | 51.40% | 5.8 ms | 172.4 FPS |
| YOLOv10m Baseline | 15.40 | 63.4 | 74.20% | 51.90% | 5.4 ms | 185.1 FPS |
| YOLOv11m Baseline | 20.04 | 67.7 | 75.10% | 52.60% | 5.0 ms | 200.0 FPS |
| **YOLOv11m + CBAM (Proposed)** | **20.07** | **67.7** | **75.70%** | **53.20%** | **4.7 ms** | **212.7 FPS** |

### Per-Class Average Precision Breakdown (AP@0.50)

| Class Tier | Vehicle Category | YOLOv11m Baseline | Proposed (YOLOv11m + CBAM) | Absolute Gain ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Dominant** | Car | 93.4% | 93.4% | Maintained |
| | CNG (Auto-Rickshaw) | 91.0% | 91.1% | +0.10% |
| | Rickshaw (Non-Motorized) | 90.5% | **90.8%** | **+0.30%** |
| **Mid-Tier** | Bus | 89.2% | 89.4% | +0.20% |
| | People (Pedestrian) | 86.6% | 86.8% | +0.20% |
| | Bike (Motorcycle) | 83.7% | 84.6% | +0.90% |
| **Minority** | Mini-Truck | 65.5% | 66.6% | +1.10% |
| | Truck | 45.5% | **47.0%** | **+1.50%** |
| | Cycle (Bicycle) | 30.5% | 31.6% | +1.10% |
| **Overall** | **Mean AP (mAP@0.50)** | **75.10%** | **75.70%** | **+0.60% Net Boost** |

---

## 📂 Repository Directory Structure

```text
├── models/
│   ├── cbam.py                 # PyTorch CBAM implementation (Channel + Spatial Attention)
│   ├── dcfm.py                 # PyTorch DCFM_v2 implementation (Multi-scale dilated fusion)
│   ├── yolov11m_cbam.yaml      # Proposed Layer-23 CBAM architecture specification
│   ├── yolov11m_dcfm.yaml      # Proposed DCFM architecture specification
│   └── data_traffic9.yaml      # 9-class dataset configuration file
├── IPYNBS/
│   ├── Ultimate_Thesis.ipynb   # Master end-to-end training notebook
│   └── ...                     # Kaggle and Colab experiment logs
├── Springer_Journal_Manuscript/
│   ├── LaTeX_Overleaf/         # Complete Springer Nature LaTeX source files
│   │   ├── main.tex            # Primary LaTeX manuscript
│   │   ├── sn-article.tex      # Overleaf entry-point alias
│   │   ├── sn-bibliography.bib # Complete 36 BibTeX references
│   │   ├── sn-jnl.cls          # Official Springer Nature class file
│   │   └── figures/            # 10 High-resolution publication figures
│   └── Springer_JRTIP_Overleaf.zip # Ready-to-upload Overleaf archive
├── app.py                      # Real-time Streamlit Edge Surveillance Prototype
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hijbullahx/Attention-Guided-YOLOv11-for-Object-Detection-in-Unstructured-Bangladeshi-Traffic.git
cd Attention-Guided-YOLOv11-for-Object-Detection-in-Unstructured-Bangladeshi-Traffic
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Real-Time Streamlit Prototype
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test real-time video ingestion, telemetry counters, and multi-class detections.

### 4. Train the Model with Custom Architecture
```python
from ultralytics import YOLO
import models.cbam  # Registers CBAM into Ultralytics namespace

# Initialize model from custom architecture YAML
model = YOLO("models/yolov11m_cbam.yaml").load("yolo11m.pt")

# Train on custom 9-class dataset
model.train(
    data="models/data_traffic9.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    project="traffic_results",
    name="yolov11m_cbam_run"
)
```

---

## 📄 Springer Nature LaTeX Manuscript (Overleaf)

The complete camera-ready LaTeX source for submission to **Springer Nature (Journal of Real-Time Image Processing)** is located in:
- Folder: `Springer_Journal_Manuscript/LaTeX_Overleaf/`
- 1-Click Zip: `Springer_Journal_Manuscript/Springer_JRTIP_Overleaf.zip`

Upload `Springer_JRTIP_Overleaf.zip` directly to [Overleaf](https://www.overleaf.com/) to compile the complete manuscript with all 10 figures and 36 citations.

---

## 📖 Citation

If you find this research work or codebase helpful in your research, please cite our paper:

```bibtex
@article{hijbullah2026realtime,
  title={Real-Time Attention-Guided Object Detection and Dynamic Feature Fusion for Dense Unstructured Traffic in Developing Regions},
  author={Hijbullah, Md. Taher Bin Omar and Mia, Md. Rony and Hasan, Mahedi and Islam, Md. Rashedul and Das, Utpal Kanti},
  journal={Journal of Real-Time Image Processing},
  publisher={Springer Nature},
  year={2026}
}
```

---

## 👥 Authors & Acknowledgments

* **Md. Taher Bin Omar Hijbullah** - Department of Computer Science and Engineering, IUBAT
* **Md. Rony Mia** - Department of Computer Science and Engineering, IUBAT
* **Mahedi Hasan** (*Corresponding Author*) - Department of Computer Science and Engineering, IUBAT (`mahedi.hasan@iubat.edu`)
* **Md. Rashedul Islam** - Department of Computer Science and Engineering, IUBAT
* **Utpal Kanti Das** - Department of Computer Science and Engineering, IUBAT

*We gratefully acknowledge the Department of Computer Science and Engineering at IUBAT—International University of Business Agriculture and Technology for providing the computational infrastructure and academic support.*

---

## 📜 License
This project is open-source and released under the [MIT License](LICENSE).
