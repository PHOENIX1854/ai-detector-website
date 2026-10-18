import os
import tempfile
import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st

import detector_logic

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI vs. Real Media Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- EXACT HTML/CSS STYLING FROM ai_detector.html ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global font & background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Background matching ai_detector.html */
    .stApp {
        background-color: #f3f4f6 !important;
    }
    
    /* Hide Streamlit chrome: header, footer, sidebar */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    
    /* Main container sizing */
    .block-container {
        max-width: 960px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* Card wrapper */
    .card-container {
        background-color: #ffffff;
        border-radius: 0.75rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    
    /* Card Header */
    .card-header {
        background-color: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 1.5rem 2rem;
    }
    .card-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
        line-height: 1.25;
    }
    .card-header p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        color: #4b5563;
        font-size: 0.95rem;
    }
    
    /* Detect Button Styling (Sky-600) */
    div.stButton > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 0.375rem !important;
        border: none !important;
        padding: 0.65rem 1rem !important;
        width: 100% !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: background-color 0.15s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
    }
    
    /* Section Headings */
    .section-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    
    /* Colorbar gradient */
    .heatmap-scale {
        height: 10px;
        border-radius: 9999px;
        background: linear-gradient(to right, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000, #800000);
        margin-top: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def data_url_to_pil(data_url: str):
    """Converts a base64 Data URL to a PIL Image."""
    if not data_url or not isinstance(data_url, str):
        return None
    try:
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None

def blend_images(orig_pil: Image.Image, heat_pil: Image.Image, alpha_pct: int) -> Image.Image:
    """Blends original and heatmap images given an alpha percentage (0-100)."""
    alpha = alpha_pct / 100.0
    orig_np = np.array(orig_pil.convert("RGB"))
    heat_np = np.array(heat_pil.convert("RGB"))
    
    if orig_np.shape[:2] != heat_np.shape[:2]:
        heat_np = cv2.resize(heat_np, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    blended = cv2.addWeighted(orig_np, 1.0 - alpha, heat_np, alpha, 0)
    return Image.fromarray(blended)

def render_image(img, caption=None):
    """Renders image compatibly across Streamlit versions."""
    try:
        st.image(img, caption=caption, use_column_width=True)
    except TypeError:
        st.image(img, caption=caption)

# --- CACHED MODEL LOADER ---
@st.cache_resource(show_spinner="Loading AI models... (This may take a moment)")
def load_models_cached():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vlm_model, processor, classifier_model = detector_logic.load_models(device)
    return vlm_model, processor, classifier_model, device

# Load models
try:
    VLM_MODEL, PROCESSOR, CLASSIFIER_MODEL, DEVICE = load_models_cached()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- CARD HEADER (EXACT ai_detector.html) ---
st.markdown("""
<div class="card-container">
    <div class="card-header">
        <h1>AI vs. Real Media Detector</h1>
        <p>Upload an image or video to see if it was generated by AI, with visual transformer attention heatmaps.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- FILE UPLOADER (EXACT ACCEPT TYPES) ---
uploaded_file = st.file_uploader(
    "Choose file",
    type=["jpeg", "jpg", "png", "webp", "mp4", "mov", "avi", "mkv", "webm"],
    help="Images (PNG, JPG, WEBP) or Videos (MP4, MOV, AVI, MKV)",
    label_visibility="collapsed"
)

# Detect Button
detect_clicked = st.button("Detect", key="detect_btn")

# Session state initialization
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_filename" not in st.session_state:
    st.session_state["last_filename"] = None
if "slider_val" not in st.session_state:
    st.session_state["slider_val"] = 50

# Preset button helpers
def set_opacity(val):
    st.session_state["slider_val"] = val

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    is_image = file_ext in detector_logic.IMAGE_EXTENSIONS_SET
    is_video = file_ext in detector_logic.VIDEO_EXTENSIONS_SET

    # If the user clicks Detect, run inference and save result in session state
    if detect_clicked:
        with st.spinner("Analyzing media... this may take a moment."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                if is_image:
                    result = detector_logic.get_prediction_for_image_file(
                        tmp_path, VLM_MODEL, PROCESSOR, CLASSIFIER_MODEL, DEVICE
                    )
                elif is_video:
                    result = detector_logic.process_video_file(
                        tmp_path, VLM_MODEL, PROCESSOR, CLASSIFIER_MODEL, DEVICE
                    )
                else:
                    result = {"error": f"Unsupported file type: {file_ext}"}
            except Exception as exc:
                result = {"error": str(exc)}
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            st.session_state["last_result"] = result
            st.session_state["last_filename"] = uploaded_file.name

    # Two columns matching md:flex md:space-x-8 from ai_detector.html
    col_preview, col_result = st.columns(2, gap="large")

    with col_preview:
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        if is_image:
            img_bytes = uploaded_file.getvalue()
            pil_preview = Image.open(BytesIO(img_bytes)).convert("RGB")
            render_image(pil_preview)
        elif is_video:
            st.video(uploaded_file)

    with col_result:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        if st.session_state["last_result"] is not None and st.session_state["last_filename"] == uploaded_file.name:
            data = st.session_state["last_result"]

            if "error" in data:
                st.error(data["error"])
            else:
                label = data.get("label", "Real")
                confidence = data.get("confidence", 0.0)
                confidence_percent = confidence * 100.0
                threshold = data.get("threshold", 0.55)
                frames_sampled = data.get("frames_sampled")
                keyframe_index = data.get("keyframe_index")
                heatmap_raw_b64 = data.get("heatmap_raw")
                orig_image_b64 = data.get("original_image")

                is_ai = label.lower() == "ai"

                # Colors matching ai_detector.js
                card_bg = "#fef2f2" if is_ai else "#ecfdf5"
                card_border = "#fecaca" if is_ai else "#a7f3d0"
                text_color = "#991b1b" if is_ai else "#065f46"
                bar_color = "#dc2626" if is_ai else "#059669"

                frames_tag = ""
                if frames_sampled:
                    key_text = f" (Key #{keyframe_index})" if keyframe_index else ""
                    frames_tag = f"<span>•</span><span>{frames_sampled} frames{key_text}</span>"

                # Circular progress meter HTML identical to ai_detector.js (no indentation to avoid markdown codeblock)
                verdict_html = (
                    f'<div style="background-color: {card_bg}; border: 1px solid {card_border}; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">'
                    f'<div style="display: flex; flex-direction: row; align-items: center; justify-content: space-between; gap: 1rem;">'
                    f'<div>'
                    f'<div style="font-size: 1.875rem; font-weight: 800; color: #111827; line-height: 1.25;">'
                    f'{confidence_percent:.1f}% <span style="font-size: 1rem; font-weight: 400; color: #6b7280;">Confidence</span>'
                    f'</div>'
                    f'<div style="display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-top: 0.5rem; font-size: 0.75rem; color: #4b5563; font-weight: 500;">'
                    f'<span style="background: rgba(255,255,255,0.9); border: 1px solid #d1d5db; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; color: #374151;">TTA Stabilized</span>'
                    f'<span>•</span>'
                    f'<span style="font-size: 11px; color: #6b7280;">Threshold: {(threshold * 100):.0f}%</span>'
                    f'{frames_tag}'
                    f'</div>'
                    f'</div>'
                    f'<div style="position: relative; width: 80px; height: 80px; flex-shrink: 0; display: flex; align-items: center; justify-content: center;">'
                    f'<svg viewBox="0 0 36 36" style="width: 100%; height: 100%; transform: rotate(-90deg);">'
                    f'<circle stroke="#e5e7eb" stroke-width="3.5" fill="transparent" r="15.9155" cx="18" cy="18" />'
                    f'<circle stroke="{bar_color}" stroke-width="3.5" stroke-linecap="round" fill="transparent" stroke-dasharray="{confidence_percent:.1f}, 100" stroke-dashoffset="0" r="15.9155" cx="18" cy="18" />'
                    f'</svg>'
                    f'<div style="position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; color: {text_color};">'
                    f'{label}'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(verdict_html, unsafe_allow_html=True)

                # Heatmap Section identical to ai_detector.js
                if heatmap_raw_b64 and orig_image_b64:
                    orig_pil = data_url_to_pil(orig_image_b64)
                    heat_pil = data_url_to_pil(heatmap_raw_b64)

                    if orig_pil and heat_pil:
                        sub_text = f"Keyframe #{keyframe_index} anomaly heatmap (49 ViT patches)" if frames_sampled else "Visualizes the 49 Vision Transformer patches driving this decision"
                        heatmap_header_html = (
                            f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">'
                            f'<div>'
                            f'<div style="font-weight: 700; color: #111827; font-size: 0.95rem;">Vision Attention Heatmap</div>'
                            f'<div style="font-size: 0.75rem; color: #6b7280; margin-top: 2px;">{sub_text}</div>'
                            f'</div>'
                            f'<span style="font-size: 0.75rem; font-weight: 600; background-color: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px;">ViT-B/32</span>'
                            f'</div>'
                        )
                        st.markdown(heatmap_header_html, unsafe_allow_html=True)

                        # Opacity Slider matching ai_detector.js
                        current_opacity = st.slider(
                            "Heatmap Overlay Intensity:",
                            min_value=0,
                            max_value=100,
                            value=st.session_state["slider_val"],
                            step=1,
                            key="heatmap_slider"
                        )
                        st.session_state["slider_val"] = current_opacity

                        # Render blended heatmap
                        blended_img = blend_images(orig_pil, heat_pil, current_opacity)
                        render_image(blended_img)

                        # Preset Quick Buttons: Original (0%), Blended (50%), Heatmap (100%)
                        b_col1, b_col2, b_col3 = st.columns(3)
                        with b_col1:
                            if st.button("Original (0%)", use_container_width=True):
                                set_opacity(0)
                                st.rerun()
                        with b_col2:
                            if st.button("Blended (50%)", use_container_width=True):
                                set_opacity(50)
                                st.rerun()
                        with b_col3:
                            if st.button("Heatmap (100%)", use_container_width=True):
                                set_opacity(100)
                                st.rerun()

                        # Color Scale Legend identical to ai_detector.js
                        legend_html = (
                            '<div style="padding-top: 0.75rem; border-top: 1px solid #e5e7eb; margin-top: 0.75rem;">'
                            '<div style="display: flex; justify-content: space-between; font-size: 11px; color: #6b7280; font-weight: 500;">'
                            '<span>Cool Blue: Neutral / Background</span>'
                            '<span>Hot Red: Peak Attention / Artifacts</span>'
                            '</div>'
                            '<div class="heatmap-scale"></div>'
                            '</div>'
                        )
                        st.markdown(legend_html, unsafe_allow_html=True)

        elif not detect_clicked and uploaded_file is not None:
            st.info("Click **Detect** to run the analysis.")
