# CNN Image Classifier API

A simple FastAPI app that deploys a CNN trained on the CIFAR-10 dataset.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
