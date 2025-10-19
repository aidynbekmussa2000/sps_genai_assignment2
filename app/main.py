# app/main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from .infer import load_model, load_labels, preprocess_image, predict

app = FastAPI(title="CNN Classifier")

model = None
device = None
labels = load_labels()

@app.get("/")
def root():
    return {
        "message": "Welcome to the CNN Image Classifier API 👋",
        "endpoints": {
            "/health": "server & model status",
            "/labels": "list of class names",
            "/predict": "upload an image (multipart/form-data)"
        }
    }

@app.on_event("startup")
def startup():
    global model, device
    weights = os.getenv("WEIGHTS_PATH", "artifacts/model.pt")
    if os.path.exists(weights):
        model, device = load_model(weights)
    else:
        print(f"WARNING: {weights} not found. /predict returns 503 until you train.")

class Out(BaseModel):
    label: str
    index: int
    confidence: float
    probs: list[float]

@app.get("/labels")
def list_labels():
    return {"labels": labels}

@app.get("/health")
def health():
    return {"status": "ok", "has_model": model is not None}

@app.post("/predict", response_model=Out)
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")
    x = preprocess_image(await file.read())
    idx, conf, probs = predict(model, device, x)
    label = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
    return Out(label=label, index=idx, confidence=conf, probs=probs)
