FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and configure Python environments
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install securely
COPY requirements.txt .
# pip will see PyTorch is already installed and skip pulling the 2GB CUDA wheels
RUN pip install --no-cache-dir -r requirements.txt

# Secure Hugging Face space constraints (Run as non-root user 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Hugging Face exposes on port 7860 by default
ENV FLASK_APP=app.py
ENV FORCE_CPU=1

EXPOSE 7860

# Execute Flask bridge natively binding to all interfaces explicitly on HuggingFace 7860
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
