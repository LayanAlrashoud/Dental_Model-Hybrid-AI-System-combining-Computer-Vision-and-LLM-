# Dental AI Analyzer

An AI-powered web application for preliminary dental X-ray analysis.  
The system allows users to upload a dental image, detects visible dental findings using a YOLO-based model, and generates a concise AI-written report in plain English.

## Overview

This project combines computer vision and large language models to support dental image analysis in a simple web interface.

The workflow is:

1. The user uploads a dental X-ray image.
2. A YOLO model analyzes the image and detects dental findings.
3. Duplicate overlapping detections are filtered using IoU-based deduplication.
4. The system summarizes detected findings.
5. An LLM generates a short, professional preliminary report.
6. The user can review:
   - the original image
   - detected findings one by one
   - a detection summary
   - AI-generated status and attention level

## Features

- Upload dental X-ray images through a Flask web interface
- Detect dental findings using a YOLO model
- Remove duplicate detections with IoU filtering
- Show all findings and also allow browsing each finding individually
- Classify the overall case into a general status
- Estimate an attention level: Low / Medium / High
- Generate a concise AI-written dental summary
- Display findings, confidence scores, and image region information

## Supported Findings

The system is currently designed to summarize the following classes:

- Caries
- Filling
- Crown
- Implant
- Periapical lesion
- Retained root
- Root canal filling

## Tech Stack

- **Backend:** Flask
- **Computer Vision:** Ultralytics YOLO
- **Image Processing:** OpenCV
- **LLM Report Generation:** OpenAI API
- **Frontend:** HTML, CSS, JavaScript, Jinja2 templates

## Project Structure

```bash
project/
│
├── app.py
├── config.py
├── uploads/
├── outputs/
│   ├── original/
│   └── predicted/
│
├── services/
│   ├── yolo_service.py
│   ├── report_service.py
│   └── prompt_builder.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── css/
