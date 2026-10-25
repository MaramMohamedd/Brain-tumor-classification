import io
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.config import FRAMEWORK, MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

app = FastAPI(title="Brain Tumor Classification API")

# متغير عام لحفظ الموديل
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # تحميل موديل TorchScript مباشرة دون الحاجة لإعادة بناء الهيكل
        model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "framework": FRAMEWORK,
        "classes": CLASS_NAMES
    }

# إعدادات الـ Normalization الخاصة بـ PyTorch
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t = torch.from_numpy(np.array(img).copy()).permute(2, 0, 1).float()[None]
    t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False, antialias=False)
    t = t.round().clamp(0, 255).to(torch.uint8)[0]
    x = t.float().div_(255)
    x = (x.unsqueeze(0) - MEAN) / STD
    return x

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    start_time = time.time()
    contents = await file.read()
    
    try:
        x = preprocess_image(contents)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            confidence, pred_idx = torch.max(probs, dim=0)
            
        pred_label = CLASS_NAMES[pred_idx.item()]
        latency = int((time.time() - start_time) * 1000)
        
        probabilities_dict = {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))}
        
        return {
            "framework": FRAMEWORK,
            "label": pred_label,
            "confidence": round(confidence.item(), 4),
            "probabilities": probabilities_dict,
            "latency_ms": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))