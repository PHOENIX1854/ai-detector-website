---
title: AI vs Real Media Detector
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# AI vs Real Media Detector

A full-stack AI media detector powered by OpenAI's CLIP (`openai/clip-vit-base-patch32`) and a custom PyTorch MLP classifier.

## Features
- **Image Detection:** Fast inference on PNG, JPG, and WEBP images.
- **Video Detection:** Samples keyframes from MP4, MOV, and AVI videos using batched frame inference.
- **Web Interface:** Clean, responsive drag-and-drop UI with confidence visualization.

## Running Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the web server:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in your browser.
