# 🦷 Dental AI Analyzer

An AI-powered web application for dental X-ray analysis that combines **Computer Vision (YOLOv8 Segmentation)** and **Large Language Models (LLMs)** to detect dental findings and generate a professional preliminary report.

---

## 🚀 Overview

This project provides an end-to-end AI system that:

1. Accepts a dental X-ray image from the user  
2. Detects dental findings using a YOLOv8 segmentation model  
3. Filters duplicate detections using IoU-based post-processing  
4. Summarizes detected findings  
5. Classifies overall case status and attention level  
6. Generates a concise AI-powered dental report  

The system simulates a real-world intelligent assistant for dental image interpretation.

---

## ✨ Features

- Upload dental X-ray images  
- Detect multiple dental conditions using YOLOv8-seg  
- Remove duplicate detections (IoU filtering)  
- Interactive viewer (browse findings one-by-one)  
- Detection summary + table  
- AI-generated dental report  
- Case classification:
  - Status
  - Attention Level (Low / Medium / High)

---

## 🧠 AI Components

### 1. Computer Vision (YOLOv8-seg)

- Model: YOLOv8m-seg  
- Performs both:
  - **Bounding Box detection (location)**
  - **Segmentation (precise shape)**
- Filters low-confidence predictions  
- Applies IoU-based deduplication  

---

### 2. LLM (Report Generator)

- Converts structured detections into natural language  
- Generates:
  - Summary  
  - Interpretation  
  - Recommendations  
- Output is:
  - Clear  
  - Concise  
  - Professional  

---

## 📊 Dataset

### Source

- Platform: Roboflow  
- Workspace: layans-workspace  
- Project: denim-dh8es  
- Version: 2  

---

### Data Statistics

#### Training Set
- Images: **4,500**
- Objects (annotations): **31,438**

#### Validation Set
- Images: **327**
- Objects: **2,323**

#### Test Set
- Images: **566**
- Objects: **3,974**

#### Removed During Cleaning
- Train: 3,000 images  
- Validation: 611 images  
- Test: 371 images  

---

### Classes

- Caries  
- Filling  
- Crown  
- Implant  
- Periapical lesion  
- Retained root  
- Root canal filling  

---

### Annotation Details

Each object represents a dental finding and consists of:

- A class label  
- A bounding box  
- A segmentation mask  

If an image contains multiple findings, each is counted separately.

---

### Preprocessing

- Removed low-quality images  
- Cleaned incorrect annotations  
- Merged similar classes  
- Converted to YOLO format  
- Ensured consistency across dataset splits  

---


