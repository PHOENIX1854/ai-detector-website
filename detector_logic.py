import sys
import os
import glob
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
SAMPLE_RATE_SECONDS = 0.5 # Sample 2 frames per second

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
    vlm_model = CLIPModel.from_pretrained(VLM_MODEL_NAME).to(device)
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

# --- 4. PREDICTION FUNCTIONS (Unchanged) ---

def run_inference_on_image(pil_image, vlm, processor, classifier, device):
    """
    Runs inference on a single PIL.Image object.
    Returns the raw probability (float).
    """
    try:
        inputs = processor(text=None, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = vlm.get_image_features(**inputs)
        
        with torch.no_grad():
            outputs = classifier(image_features)
            probs = torch.sigmoid(outputs)
            prediction_num = probs.item()
        
        return prediction_num
    except Exception as e:
        print(f"\nWarning: Error during frame inference: {e}")
        return 0.0

# --- 5. UPDATED PREDICTION FUNCTIONS (Return JSON) ---

def get_prediction_for_image_file(image_path, vlm, processor, classifier, device):
    """
    Runs the full pipeline for a single *image file*
    and returns a dictionary (JSON).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        prediction_num = run_inference_on_image(img, vlm, processor, classifier, device)
        
        is_ai = prediction_num > 0.5
        label = "AI" if is_ai else "Real"
        confidence = prediction_num if is_ai else 1 - prediction_num
        
        # Return a dictionary
        return {
            "label": label,
            "confidence": confidence,
            "filename": os.path.basename(image_path)
        }
        
    except Exception as e:
        return {"error": f"Could not process file. {e}"}

def process_video_file(video_path, vlm, processor, classifier, device):
    """
    (UPDATED) Runs the full pipeline for a *video file*.
    Returns a dictionary (JSON).
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file."}

        frame_probabilities = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0 or fps is None: 
            print(f"\nWarning: Could not get FPS for {video_path}. Assuming 30.")
            fps = 30
            
        frame_skip = int(round(fps * SAMPLE_RATE_SECONDS))
        if frame_skip == 0: frame_skip = 1
            
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    prob = run_inference_on_image(pil_img, vlm, processor, classifier, device)
                    frame_probabilities.append(prob)
                except Exception as frame_e:
                    print(f"\nWarning: Skipped a frame in {video_path} due to error: {frame_e}")

            frame_count += 1
        
        cap.release()

        if not frame_probabilities:
            return {"error": "Video was empty or no frames could be read."}

        max_prob = np.max(frame_probabilities)
        is_ai = max_prob > 0.5
        label = "AI" if is_ai else "Real"
        confidence = max_prob if is_ai else 1 - max_prob
        
        # Return a dictionary
        return {
            "label": label,
            "confidence": confidence,
            "frames_sampled": len(frame_probabilities),
            "filename": os.path.basename(video_path)
        }

    except Exception as e:
        return {"error": f"Could not process video. {e}"}
