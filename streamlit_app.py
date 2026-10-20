
import os
import inspect
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
        color: #4b5663;
        font-size: 0.95rem;
    }
    
    /* Detect Button Styling (Sky-600) */
    div.stButton, div[data-testid="stButton"] {
        width: 100% !important;
    }
    div.stButton > button, div[data-testid="stButton"] > button {
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
    div.stButton > button:hover, div[data-testid="stButton"] > button:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
    }
    div.stButton > button:disabled, div[data-testid="stButton"] > button:disabled {
        background-color: #d1d5db !important;
        color: #9ca3af !important;
        cursor: not-allowed !important;
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

def extract_confidence_pct(data: dict):
    """Extracts and normalizes confidence/probability percentage (0-100) from result dict."""
    candidate_keys = [
        "confidence", "probability", "prob", "score", 
        "ai_probability", "real_probability", "confidence_score", "pct", "percentage"
    ]
    raw_val = None
    for key in candidate_keys:
        if key in data and data[key] is not None:
            raw_val = data[key]
            break

    if raw_val is None:
        return None

    try:
        if isinstance(raw_val, str):
            raw_val = raw_val.replace("%", "").strip()
        val = float(raw_val)
        # Convert 0.0-1.0 float probabilities into 0-100 percentages
        if 0.0 <= val <= 1.0:
            val *= 100.0
        return max(0.0, min(100.0, val))
    except (ValueError, TypeError):
        return None

def render_image(img, caption=None):
    """Renders image compatibly across Streamlit versions without type checker or runtime errors."""
    params = inspect.signature(st.image).parameters
    kwargs = {}
    if "use_container_width" in params:
        kwargs["use_container_width"] = True
    elif "use_column_width" in params:
        kwargs["use_column_width"] = True
    
    try:
        st.image(img, caption=caption, **kwargs)
    except TypeError:
        try:
            # Fallback for Streamlit >= 1.41+ where width="stretch" is used
            st.image(img, caption=caption, width="stretch")
        except TypeError:
            st.image(img, caption=caption)

def full_width_button(label, key=None, disabled=False):
    """Renders a full-width button compatibly across Streamlit versions."""
    params = inspect.signature(st.button).parameters
    kwargs = {}
    if "use_container_width" in params:
        kwargs["use_container_width"] = True

    try:
        return st.button(label, key=key, disabled=disabled, **kwargs)
    except TypeError:
        return st.button(label, key=key, disabled=disabled)

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

    # Reset result if a new file is uploaded
    if st.session_state["last_filename"] != uploaded_file.name:
        st.session_state["last_result"] = None
        st.session_state["last_filename"] = uploaded_file.name
        st.session_state["slider_val"] = 50

    has_results = (
        st.session_state["last_result"] is not None and 
        "error" not in st.session_state["last_result"]
    )

    # 1. Preview ABOVE the Detect button (Heatmap overlays directly on preview when analyzed)
    if has_results and is_image:
        res = st.session_state["last_result"]
        orig_pil = data_url_to_pil(res.get("original_image"))
        heat_pil = data_url_to_pil(res.get("heatmap_raw"))

        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 1rem; margin-bottom: 0.5rem;">
            <div class="section-title" style="margin: 0;">Preview & Vision Attention Heatmap</div>
            <span style="font-size: 0.75rem; font-weight: 600; background-color: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px;">ViT-B/32</span>
        </div>
        """, unsafe_allow_html=True)

        current_opacity = st.slider(
            "Heatmap Overlay Intensity:",
            min_value=0,
            max_value=100,
            value=st.session_state["slider_val"],
            step=1,
            key="heatmap_slider"
        )
        st.session_state["slider_val"] = current_opacity

        # Overlay heatmap directly on preview
        if orig_pil and heat_pil:
            blended_preview = blend_images(orig_pil, heat_pil, current_opacity)
            render_image(blended_preview)
        elif orig_pil:
            render_image(orig_pil)

        # Preset Quick Buttons
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            if full_width_button("Original (0%)", key="preset_orig_img"):
                set_opacity(0)
                st.rerun()
        with b_col2:
            if full_width_button("Blended (50%)", key="preset_blend_img"):
                set_opacity(50)
                st.rerun()
        with b_col3:
            if full_width_button("Heatmap (100%)", key="preset_heat_img"):
                set_opacity(100)
                st.rerun()

    elif has_results and is_video:
        res = st.session_state["last_result"]
        orig_pil = data_url_to_pil(res.get("original_image"))
        heat_pil = data_url_to_pil(res.get("heatmap_raw"))
        keyframe_index = res.get("keyframe_index", 1)

        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 1rem; margin-bottom: 0.5rem;">
            <div class="section-title" style="margin: 0;">Keyframe #{keyframe_index} Attention Heatmap</div>
            <span style="font-size: 0.75rem; font-weight: 600; background-color: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px;">ViT-B/32</span>
        </div>
        """, unsafe_allow_html=True)

        current_opacity = st.slider(
            "Heatmap Overlay Intensity:",
            min_value=0,
            max_value=100,
            value=st.session_state["slider_val"],
            step=1,
            key="heatmap_slider"
        )
        st.session_state["slider_val"] = current_opacity

        if orig_pil and heat_pil:
            blended_preview = blend_images(orig_pil, heat_pil, current_opacity)
            render_image(blended_preview)

        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            if full_width_button("Original (0%)", key="preset_orig_vid"):
                set_opacity(0)
                st.rerun()
        with b_col2:
            if full_width_button("Blended (50%)", key="preset_blend_vid"):
                set_opacity(50)
                st.rerun()
        with b_col3:
            if full_width_button("Heatmap (100%)", key="preset_heat_vid"):
                set_opacity(100)
                st.rerun()

        with st.expander("🎥 View Original Video"):
            st.video(uploaded_file)

    else:
        # Before detection: show standard preview
        st.markdown('<div class="section-title" style="margin-top: 1rem;">Preview</div>', unsafe_allow_html=True)
        if is_image:
            img_bytes = uploaded_file.getvalue()
            pil_preview = Image.open(BytesIO(img_bytes)).convert("RGB")
            render_image(pil_preview)
        elif is_video:
            st.video(uploaded_file)

    # 2. Horizontal Results Display (ABOVE the Detect button)
    if has_results:
        data = st.session_state["last_result"]

        st.write("")
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        col_verdict, col_details = st.columns([1, 1], gap="large")

        label = data.get("label", "Real")
        frames_sampled = data.get("frames_sampled")
        keyframe_index = data.get("keyframe_index")
        confidence_pct = extract_confidence_pct(data)

        is_ai = label.lower() == "ai"
        display_label = "AI" if is_ai else "REAL"

        # Simple Red box for AI, Green box for REAL
        box_bg = "#dc2626" if is_ai else "#16a34a"
        box_border = "#b91c1c" if is_ai else "#15803d"
        card_bg = "#fef2f2" if is_ai else "#f0fdf4"
        card_border = "#fecaca" if is_ai else "#bbf7d0"
        text_accent = "#991b1b" if is_ai else "#166534"

        # Build HTML for confidence percentage + progress bar if available
        confidence_html = ""
        if confidence_pct is not None:
            confidence_html = (
                f'<div style="margin-top: 0.85rem; width: 100%; max-width: 220px;">'
                f'<div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 700; color: {text_accent}; margin-bottom: 0.3rem;">'
                f'<span>Confidence</span>'
                f'<span>{confidence_pct:.1f}%</span>'
                f'</div>'
                f'<div style="width: 100%; background-color: rgba(0,0,0,0.08); height: 8px; border-radius: 9999px; overflow: hidden;">'
                f'<div style="width: {confidence_pct:.1f}%; background-color: {box_bg}; height: 100%; border-radius: 9999px;"></div>'
                f'</div>'
                f'</div>'
            )

        with col_verdict:
            verdict_html = (
                f'<div style="background-color: {card_bg}; border: 1px solid {card_border}; padding: 1.25rem 1.5rem; border-radius: 0.75rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem; min-height: 155px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">'
                f'<div style="background-color: {box_bg}; border: 2px solid {box_border}; border-radius: 0.5rem; padding: 0.6rem 2.75rem; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">'
                f'<span style="font-size: 2rem; font-weight: 900; color: #ffffff; letter-spacing: 0.1em; line-height: 1;">'
                f'{display_label}'
                f'</span>'
                f'</div>'
                f'{confidence_html}'
                f'</div>'
            )
            st.markdown(verdict_html, unsafe_allow_html=True)

        with col_details:
            details_html = (
                '<div style="background-color: #ffffff; border: 1px solid #e5e7eb; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem; min-height: 155px; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">'
                '<div style="font-weight: 700; color: #111827; font-size: 0.95rem;">Vision Transformer Heatmap Scale</div>'
                '<span style="font-size: 0.75rem; font-weight: 600; background-color: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px;">ViT-B/32</span>'
                '</div>'
                '<div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.75rem;">'
                + ("Keyframe peak anomaly attention map (49 ViT patches)." if frames_sampled else "Heatmap overlay indicates the 49 spatial ViT patches driving the prediction.") +
                '</div>'
                '</div>'
                '<div>'
                '<div style="display: flex; justify-content: space-between; font-size: 11px; color: #6b7280; font-weight: 500;">'
                '<span>Cool Blue: Neutral / Background</span>'
                '<span>Hot Red: Peak Attention / Artifacts</span>'
                '</div>'
                '<div class="heatmap-scale"></div>'
                '</div>'
                '</div>'
            )
            st.markdown(details_html, unsafe_allow_html=True)

    elif st.session_state["last_result"] is not None and "error" in st.session_state["last_result"]:
        st.error(st.session_state["last_result"]["error"])

    st.write("")

    # 3. Detect Button (BELOW Preview and Results) - full width
    detect_clicked = full_width_button("Detect", key="detect_btn")

    # If Detect clicked, run inference
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
            st.rerun()

else:
    # When no file uploaded, show disabled Detect button - full width
    full_width_button("Detect", key="detect_btn_disabled", disabled=True)
