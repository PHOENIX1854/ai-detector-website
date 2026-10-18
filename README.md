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

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.

## Deploying to Streamlit Community Cloud (streamlit.io)
1. Push your repository to GitHub (`PHOENIX1854/ai-detector-website`).
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New app**.
4. Select:
   - **Repository:** `PHOENIX1854/ai-detector-website`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **Deploy!**

