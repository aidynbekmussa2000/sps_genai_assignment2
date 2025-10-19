# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .infer import load_model, load_labels, preprocess_image, predict
import os

app = FastAPI(title="CNN Classifier")

model = None
device = None
labels = None

@app.get("/")
def root():
    return {
        "message": "Welcome to the CNN Image Classifier API 👋",
        "endpoints": {"/health": "server & model status", "/predict": "upload an image (multipart/form-data)"}
    }

@app.on_event("startup")
def startup():
    global model, device, labels
    labels = load_labels()
    weights = "artifacts/model.pt"
    if not os.path.exists(weights):
        print(f"WARNING: {weights} not found. /predict will return 503 until you train.")
    else:
        model, device = load_model(weights)

class Out(BaseModel):
    label: str
    index: int
    confidence: float
    probs: list

@app.get("/health")
def health():
    return {"status": "ok", "has_model": model is not None}

@app.post("/predict", response_model=Out)
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file.")
    x = preprocess_image(await file.read())
    idx, conf, probs = predict(model, device, x)
    label = labels[idx] if idx < len(labels) else f"class_{idx}"
    return Out(label=label, index=idx, confidence=conf, probs=probs)