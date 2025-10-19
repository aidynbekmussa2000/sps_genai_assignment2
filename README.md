# CNN Image Classifier API

A simple FastAPI app that deploys a CNN trained on the CIFAR-10 dataset.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

# Docker

```bash
docker build -t cnn-api .
docker run --rm -p 8000:8000 cnn-api
# with trained weights mounted (optional):
# docker run --rm -p 8000:8000 -e WEIGHTS_PATH=/weights/model.pt -v $PWD/artifacts:/weights cnn-api
```

# Smoke test
```bash
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/predict" -F "image=@sample_images/example.jpg"
```
