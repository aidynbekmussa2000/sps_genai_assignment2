# Use a lightweight Python base image
FROM python:3.11-slim

# Install required system libraries (needed for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (for safety)
RUN useradd -m appuser

# Set the working directory
WORKDIR /app

# Copy dependency file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the image
COPY app ./app
COPY artifacts ./artifacts

# Expose FastAPI port
EXPOSE 8000

# Health check endpoint (Docker will use this to confirm the app is running)
HEALTHCHECK --interval=30s --timeout=3s CMD curl --fail http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Run FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]