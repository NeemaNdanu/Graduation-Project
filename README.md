# Real-Time Traffic Detection & Speed Estimation — Kenya
 
A computer vision system that detects road users and estimates vehicle speeds in real time from traffic video footage, built as a final year project at USIU-Africa.
 
**Live App:** `Traffic Intelligence App`&nbsp;|&nbsp; **Thesis:** `4900VA_NeemaNdanu_FinalProject.pdf`
 
---
 
## Overview
 
Road traffic accidents remain a critical public health challenge in Kenya, with over 2,400 fatalities recorded in the first half of 2024 alone (NTSA, 2024). This project addresses the lack of real-time monitoring tools by building an AI-driven system that detects accidents, classifies road users, and estimates vehicle speeds, using publicly available video footage from the X platform (formerly Twitter).
 
---
 
## Repository Structure
 
| File / Folder | Description |
|---|---|
| `streamlit_app.py` | Main app entry point, ie., upload a video and view detections live |
| `video_detect.py` | Object detection pipeline using YOLOv8 |
| `speed_estimation.py` | Centroid-based vehicle speed tracking |
| `combined_mode.py` | Ensemble detection combining both trained models |
| `models/` | Trained YOLOv8 model weights |
| `videos/` | Placeholder folder for test video inputs |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System-level packages for Streamlit Cloud |
| `4900VA_NeemaNdanu_FinalProject.pdf` | Full research thesis |
 
---
 
## How It Works
 
1. A traffic video is uploaded via the Streamlit interface
2. YOLOv8 detects and classifies road users frame by frame
3. A centroid-based tracking algorithm computes vehicle displacement across frames to estimate speed (km/h)
4. Annotated output is rendered in real time with bounding boxes, class labels, and speed overlays
 
---
 
## Model Performance
 
Two YOLOv8 models were trained on a custom Kenyan traffic dataset and combined into an ensemble system. The ensemble approach improved overall reliability and reduced false positives across diverse traffic scenes. Notably, accident detection was the strongest performing category, with the system correctly identifying collision events in nearly all test cases.
 
---
 
## Setup
 
```bash
git clone https://github.com/NeemaNdanu/Traffic-Video-Analysis-with-YOLO.git
cd Traffic-Video-Analysis-with-YOLO
pip install -r requirements.txt
streamlit run streamlit_app.py
```
 
---
 
## License
 
BSD-3-Clause. For academic and research purposes. Please credit appropriately if reusing any part of this work.
