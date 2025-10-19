import io
import json
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model import SimpleCNN

# Simple resize -> tensor. Keep this the same as training.
TFM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def load_labels(path: str = "app/labels.json") -> List[str]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return [f"class_{i}" for i in range(10)]

def load_model(weights_path: str = "artifacts/model.pt", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval().to(device)
    return model, device

def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """Convert uploaded bytes -> (1,3,64,64) tensor."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    x = TFM(img).unsqueeze(0)
    return x

@torch.inference_mode()
def predict(model, device, x: torch.Tensor) -> Tuple[int, float, list[float]]:
    """Return (class_index, confidence, probabilities)."""
    x = x.to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    idx = max(range(len(probs)), key=lambda i: probs[i])
    return idx, float(probs[idx]), probs