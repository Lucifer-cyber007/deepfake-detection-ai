import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import load_model
import os
import base64
import io

# Configuration from model cfg
IMG_SIZE = 300
# Resolve model path relative to the project root so it works on any machine.
# Allows override via the DEEPFAKE_MODEL_PATH environment variable.
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'model',
    'DEEPFAKE_DETECTION_AI.pth',
)
MODEL_PATH = os.environ.get('DEEPFAKE_MODEL_PATH', _DEFAULT_MODEL_PATH)
NUM_FRAMES = 16  # For the temporal model forward pass

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = load_model(MODEL_PATH, device=device)
    return model

def extract_frames(video_path, num_frames=16):
    """
    Extracts num_frames evenly from the video at video_path.
    Returns (tensors, timestamps, thumbnails_as_base64)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        cap.release()
        return None, None, None
        
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames_tensors = []
    thumbnails = []
    timestamps = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        ts = idx / fps if fps > 0 else 0
        timestamps.append(ts)
        
        # Capture thumbnail for suspicious frames later, but let's pre-encode a tiny version
        small_frame = cv2.resize(frame, (100, 100))
        _, buffer = cv2.imencode('.jpg', small_frame)
        thumbnails.append(base64.b64encode(buffer).decode('utf-8'))

        # Preprocess for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        tensor = preprocess(img)
        frames_tensors.append(tensor)
        
    cap.release()
    if not frames_tensors:
        return None, None, None
        
    return torch.stack(frames_tensors), timestamps, thumbnails

def predict_video(video_path):
    """
    High-resolution video analysis.
    Returns overall score + per-frame heatmap data + suspicious thumbnails.
    """
    # High resolution sampling for the heatmap (up to 50 samples)
    SAMPLES = 50 
    tensors, timestamps, thumbnails = extract_frames(video_path, num_frames=SAMPLES)
    
    if tensors is None:
        return {"error": "Could not extract frames."}

    m = get_model()
    frame_results = []
    
    # We need to run inference. The model expects (B, T, C, H, W) where T=16.
    # To get per-frame scores, we can run them individually or in chunks.
    # Given 'temporal_head: mean', running individually is valid for frame-level features.
    
    with torch.no_grad():
        # Process in chunks of 16 to respect the model's design or just individually for heatmap
        # For simplicity and per-frame scores, we'll run 1x1xCxHxW
        # We'll reshape to (N, 1, C, H, W)
        inputs = tensors.unsqueeze(1).to(device) # (N, 1, 3, 300, 300)
        
        scores = []
        for i in range(inputs.shape[0]):
            logit = m(inputs[i:i+1])
            prob = torch.sigmoid(logit).item()
            scores.append(prob)
            
            frame_results.append({
                "timestamp": timestamps[i],
                "score": prob,
                "thumbnail": thumbnails[i] if prob > 0.4 else None # Only send thumbnails for relevant detections
            })

    overall_score = sum(scores) / len(scores)
    
    return {
        "overall_score": overall_score,
        "label": "FAKE" if overall_score > 0.5 else "REAL",
        "confidence": overall_score if overall_score > 0.5 else 1 - overall_score,
        "heatmap": [{"ts": t, "val": s} for t, s in zip(timestamps, scores)],
        "suspicious_frames": [f for f in frame_results if f["thumbnail"] is not None]
    }

def predict_image(image_bytes):
    """
    Takes image bytes, runs inference, and returns prediction score.
    """
    img = Image.open(image_bytes).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    m = get_model()
    with torch.no_grad():
        # Wrap as (1, 1, C, H, W) to match temporal model expectations if needed
        output = m(input_tensor.unsqueeze(1))
        prob = torch.sigmoid(output).item()
        
    return {
        "score": prob,
        "label": "FAKE" if prob > 0.5 else "REAL",
        "confidence": prob if prob > 0.5 else 1 - prob
    }
