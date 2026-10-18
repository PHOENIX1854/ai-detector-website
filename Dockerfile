FROM python:3.10-slim

# Install system dependencies required by OpenCV and media processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000 (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download CLIP model to cache during build for instant container startup
RUN python -c "from transformers import CLIPProcessor, CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Copy all application files
COPY --chown=user . .

# Ensure the uploads directory exists and is writable
RUN mkdir -p uploads

# Hugging Face Spaces listens on port 7860 by default
EXPOSE 7860

# Run with Gunicorn: 1 worker with 4 threads for low RAM usage and high responsiveness
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
