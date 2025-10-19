# Use a lightweight Python base image
FROM python:3.11-slim

# OS deps (Pillow etc.) + curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY app ./app

# Ensure artifacts dir exists even if not in repo
RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s CMD curl --fail http://localhost:8000/health || exit 1

# Drop privileges
USER appuser

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
