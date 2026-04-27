import os
import torch

# ============================================================
# Base directory for the project
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Data directory (dataset goes in ./sample/)
# ============================================================
DATASET_PATH = os.path.join(BASE_DIR, 'sample')

# ============================================================
# Vehicle types / Root folders in the dataset
# ============================================================
VEHICLE_TYPES = ['Four Wheeler', 'Large Vehicle', 'Three Wheeler', 'Two Wheeler']

# ============================================================
# Shared classes for object detection (4 classes)
# ============================================================
CLASSES = ["Four_Wheeler", "Large_Vehicle", "Three_Wheeler", "Two_Wheeler"]

# ============================================================
# YOLO Config
# ============================================================
YOLO_MODEL = 'yolov11s.pt'
DATA_YAML_PATH = os.path.join(DATASET_PATH, 'data.yaml')

# ============================================================
# Faster R-CNN Config
# ============================================================
RCNN_MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'fasterrcnn_car_detector.pth')
NUM_CLASSES = 5  # 1 (background) + 4 (vehicles)

# ============================================================
# Hardware Detection (CPU / GPU)
# ============================================================
# Set environment variable FORCE_CPU=1 to override GPU detection
# Example (PowerShell):  $env:FORCE_CPU="1"; python train_rcnn.py
# Example (cmd):         set FORCE_CPU=1 && python train_rcnn.py

def detect_device():
    """Auto-detect the best available device (GPU or CPU)."""
    force_cpu = os.environ.get('FORCE_CPU', '0') == '1'

    if force_cpu:
        print("System: FORCE_CPU=1 is set. Using CPU.")
        return 'cpu'

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"System: GPU detected — {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"System: Using CUDA (GPU) for training.")
        return 'cuda'
    else:
        print("System: No GPU detected. Using CPU for training.")
        print("System: Training will be slower on CPU. For GPU support,")
        print("        install PyTorch with CUDA from https://pytorch.org/get-started/locally/")
        return 'cpu'

DEVICE = detect_device()

# Ultralytics YOLO uses '0' for first GPU, 'cpu' for CPU
YOLO_DEVICE = '0' if DEVICE == 'cuda' else 'cpu'
