import sys
import os
import glob
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import cv2  # Import OpenCV
import math # For frame calculations

# --- 1. CONFIGURATION ---

INPUT_DIM = 512
SAMPLE_RATE_SECONDS = 0.5 # Sample up to 2 frames per second
MAX_FRAMES_PER_VIDEO = 20 # Cap maximum frames to keep video analysis fast (<3s on CPU)

# Overfitting Mitigation:
# 1. Configurable decision threshold (scores >= AI_THRESHOLD flagged as AI; reduces false positives)
AI_THRESHOLD = 0.55
# 2. Test-Time Augmentation (averages original + horizontally flipped image to eliminate orientation bias)
USE_TTA = True

# --- IMPORTANT CHANGE ---
# The server will look for this file in the SAME directory.
# Make sure you copy 'best_nn_classifier_FINAL.pth' here.
CLASSIFIER_WEIGHTS_PATH = 'best_nn_classifier_FINAL.pth'
VLM_MODEL_NAME = 'openai/clip-vit-base-patch32'

# Helper sets for the server to check file types
IMAGE_EXTENSIONS_SET = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS_SET = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


# --- 2. DEFINE THE NEURAL NETWORK ---
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# --- 3. NEW: MODEL LOADING FUNCTION ---
def load_models(device):
    """
    Loads and returns the VLM, processor, and classifier models.
    """
    print(f"Loading VLM: {VLM_MODEL_NAME}...")
    vlm_model = CLIPModel.from_pretrained(VLM_MODEL_NAME, attn_implementation="eager").to(device)
    processor = CLIPProcessor.from_pretrained(VLM_MODEL_NAME)
    vlm_model.eval()

    print(f"Loading trained classifier: {CLASSIFIER_WEIGHTS_PATH}...")
    if not os.path.exists(CLASSIFIER_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights file not found: {CLASSIFIER_WEIGHTS_PATH}. "
            "Please make sure 'best_nn_classifier_FINAL.pth' is in the same directory as app.py."
        )
        
    classifier_model = MLPClassifier(input_size=INPUT_DIM).to(device)
    classifier_model.load_state_dict(
        torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=device)
    )
    classifier_model.eval()
    print("All models loaded successfully.")
    
    return vlm_model, processor, classifier_model

# --- 4. PREDICTION FUNCTIONS WITH TTA (TEST-TIME AUGMENTATION) ---

def run_inference_on_image(pil_image, vlm, processor, classifier, device, use_tta=USE_TTA):
    """
    Runs inference on a single PIL.Image object.
    When use_tta is True, applies Test-Time Augmentation by evaluating both
    the original and horizontally flipped image, then averaging their predictions.
    Returns the raw probability (float).
    """
    try:
        if use_tta:
            img_flipped = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            batch_images = [pil_image, img_flipped]
        else:
            batch_images = [pil_image]

        inputs = processor(text=None, images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = vlm.get_image_features(**inputs)
            outputs = classifier(image_features)
            probs = torch.sigmoid(outputs).squeeze(-1)
            
            if probs.ndim == 0:
                prediction_num = float(probs.item())
            else:
                prediction_num = float(probs.mean().item())
        
        return prediction_num
    except Exception as e:
        print(f"\nWarning: Error during frame inference: {e}")
        return 0.0

def run_inference_on_batch(pil_images, vlm, processor, classifier, device, use_tta=USE_TTA):
    """
    Runs batch inference on a list of PIL.Image objects.
    When use_tta is True, batches each image along with its horizontal flip.
    Significantly faster than one-by-one inference on CPU/GPU.
    """
    if not pil_images:
        return []
    try:
        if use_tta:
            eval_images = []
            for img in pil_images:
                eval_images.extend([img, img.transpose(Image.FLIP_LEFT_RIGHT)])
        else:
            eval_images = pil_images

        inputs = processor(text=None, images=eval_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = vlm.get_image_features(**inputs)
            outputs = classifier(image_features)
            probs = torch.sigmoid(outputs).squeeze(-1)
            
            if probs.ndim == 0:
                raw_list = [probs.item()]
            else:
                raw_list = probs.cpu().numpy().tolist()

        if use_tta:
            # Average consecutive pairs (original + horizontally flipped)
            averaged_probs = []
            for i in range(0, len(raw_list), 2):
                avg = (raw_list[i] + raw_list[i + 1]) / 2.0
                averaged_probs.append(float(avg))
            return averaged_probs
        else:
            return [float(p) for p in raw_list]

    except Exception as e:
        print(f"\nWarning: Error during batch inference: {e}. Falling back to sequential inference.")
        return [run_inference_on_image(img, vlm, processor, classifier, device, use_tta=use_tta) for img in pil_images]

# --- 5. ATTENTION HEATMAP GENERATOR ---

def generate_attention_heatmap(pil_image, vlm, processor, device):
    """
    Computes an Attention Rollout heatmap across all 12 ViT layers.
    Maps which of the 49 image patches had the highest influence on the model.
    Returns:
        A dict with data URLs for blended heatmap, raw heatmap, and original image,
        or None if generation fails.
    """
    try:
        inputs = processor(text=None, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            vision_outputs = vlm.vision_model(**inputs, output_attentions=True)
            attentions = vision_outputs.attentions

        if not attentions:
            return None

        # Rollout across all 12 layers
        # Each layer attention matrix has shape [1, num_heads, 50, 50]
        rollout = torch.eye(50, device=device)
        for layer_attn in attentions:
            attn_heads_avg = layer_attn[0].mean(dim=0)
            attn_with_residual = 0.5 * attn_heads_avg + 0.5 * torch.eye(50, device=device)
            attn_normalized = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(attn_normalized, rollout)

        # Extract attention from CLS token (index 0) to 49 patches (indices 1 to 50)
        mask = rollout[0, 1:].reshape(7, 7).cpu().numpy()
        mask_min = float(mask.min())
        mask_max = float(mask.max())
        mask_norm = (mask - mask_min) / (mask_max - mask_min + 1e-8)

        # Cap display size to 800px max dimension for fast network transmission
        orig_w, orig_h = pil_image.size
        max_dim = 800
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            display_w = max(1, int(orig_w * scale))
            display_h = max(1, int(orig_h * scale))
        else:
            display_w, display_h = orig_w, orig_h

        pil_resized = pil_image.resize((display_w, display_h), Image.Resampling.BILINEAR)
        orig_cv = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)

        # Upsample 7x7 attention mask to display size with smooth bicubic interpolation
        heatmap_resized = cv2.resize(mask_norm, (display_w, display_h), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

        # Apply JET colormap (blue = low attention, green/yellow = medium, red = peak attention)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blended image (55% original photo + 45% heatmap overlay)
        blended = cv2.addWeighted(orig_cv, 0.55, heatmap_color, 0.45, 0)

        # Encode to JPEG
        _, buf_blended = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_heat = cv2.imencode('.jpg', heatmap_color, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_orig = cv2.imencode('.jpg', orig_cv, [cv2.IMWRITE_JPEG_QUALITY, 85])

        blended_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf_blended).decode('utf-8')
        raw_heatmap_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf_heat).decode('utf-8')
        original_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf_orig).decode('utf-8')

        return {
            "heatmap_blended": blended_b64,
            "heatmap_raw": raw_heatmap_b64,
            "original_image": original_b64
        }
    except Exception as e:
        print(f"\nWarning: Error generating attention heatmap: {e}")
        return None

# --- 6. PREDICTION FUNCTIONS WITH EXPLANATIONS & HEATMAP ---

def get_prediction_for_image_file(image_path, vlm, processor, classifier, device):
    """
    Runs the full pipeline for a single *image file*
    and returns a dictionary (JSON) with label, confidence, heatmap, and explanation.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        prediction_num = run_inference_on_image(img, vlm, processor, classifier, device)
        
        is_ai = prediction_num >= AI_THRESHOLD
        label = "AI" if is_ai else "Real"
        confidence = float(prediction_num if is_ai else 1 - prediction_num)
        
        # Generate the visual attention heatmap
        heatmap_data = generate_attention_heatmap(img, vlm, processor, device)
        
        if is_ai:
            explanation = (
                f"The model detected synthetic patterns with {confidence * 100:.1f}% confidence "
                f"(exceeding the {AI_THRESHOLD * 100:.0f}% AI threshold with TTA verification). "
                "The warm/red zones in the heatmap highlight the specific image patches that triggered the strongest AI signals."
            )
        else:
            explanation = ""

        result = {
            "label": label,
            "confidence": confidence,
            "raw_probability": float(prediction_num),
            "threshold": AI_THRESHOLD,
            "tta_enabled": USE_TTA,
            "filename": os.path.basename(image_path),
            "explanation": explanation
        }
        if heatmap_data:
            result.update(heatmap_data)
            
        return result
        
    except Exception as e:
        return {"error": f"Could not process file. {e}"}

def process_video_file(video_path, vlm, processor, classifier, device):
    """
    Runs the full pipeline for a *video file*.
    Samples up to MAX_FRAMES_PER_VIDEO frames, runs batched inference with TTA,
    and generates an attention heatmap for the key frame with the strongest signal.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file."}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None: 
            print(f"\nWarning: Could not get FPS for {video_path}. Assuming 30.")
            fps = 30
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and total_frames > MAX_FRAMES_PER_VIDEO:
            # Spread frames across the entire video duration
            ideal_skip = max(1, total_frames // MAX_FRAMES_PER_VIDEO)
            time_skip = int(round(fps * SAMPLE_RATE_SECONDS)) if fps > 0 else 15
            frame_skip = max(ideal_skip, time_skip)
        else:
            frame_skip = int(round(fps * SAMPLE_RATE_SECONDS))
            if frame_skip == 0:
                frame_skip = 1
            
        sampled_pil_images = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    sampled_pil_images.append(pil_img)
                    
                    if len(sampled_pil_images) >= MAX_FRAMES_PER_VIDEO:
                        break
                except Exception as frame_e:
                    print(f"\nWarning: Skipped a frame in {video_path} due to error: {frame_e}")

            frame_count += 1
        
        cap.release()

        if not sampled_pil_images:
            return {"error": "Video was empty or no frames could be read."}

        # Run fast batched inference with TTA
        frame_probabilities = run_inference_on_batch(
            sampled_pil_images, vlm, processor, classifier, device
        )

        if not frame_probabilities:
            return {"error": "Failed to analyze video frames."}

        # Identify the keyframe with the peak probability (most suspicious frame)
        max_idx = int(np.argmax(frame_probabilities))
        max_prob = float(frame_probabilities[max_idx])
        is_ai = max_prob >= AI_THRESHOLD
        label = "AI" if is_ai else "Real"
        confidence = float(max_prob if is_ai else 1 - max_prob)
        
        # Generate heatmap for the key frame
        best_frame = sampled_pil_images[max_idx]
        heatmap_data = generate_attention_heatmap(best_frame, vlm, processor, device)

        if is_ai:
            explanation = (
                f"Video flagged as AI-generated based on peak detection in sampled frame #{max_idx + 1} "
                f"({max_prob * 100:.1f}% AI probability, threshold {AI_THRESHOLD * 100:.0f}%). "
                "The heatmap highlights the visual focus areas in that key frame."
            )
        else:
            explanation = ""

        result = {
            "label": label,
            "confidence": confidence,
            "raw_probability": max_prob,
            "threshold": AI_THRESHOLD,
            "tta_enabled": USE_TTA,
            "frames_sampled": len(frame_probabilities),
            "keyframe_index": max_idx + 1,
            "filename": os.path.basename(video_path),
            "explanation": explanation
        }
        if heatmap_data:
            result.update(heatmap_data)

        return result

    except Exception as e:
        return {"error": f"Could not process video. {e}"}
