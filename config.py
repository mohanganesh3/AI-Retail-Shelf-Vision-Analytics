"""
Configuration settings for the Retail AI Pipeline
"""

import os

# === PATHS ===
YOLO_MODEL_PATH = "yolov10s_sku110k_best.pt"  # Use relative path
EMBEDDINGS_PATH = "product_embeddings.pth"  # Path to trained embeddings
REFERENCE_DIR = "clean_sku_dataset"
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
CROPS_DIR = "crops"
TEMP_DIR = "temp"  # Directory for temporary files

# === MODEL SETTINGS ===
YOLO_CONFIDENCE = 0.25  # Lower confidence threshold for testing
DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"

# === CAQE PARAMETERS ===
SPATIAL_SIGMA = 0.3

# === GAP DETECTION PARAMETERS ===
X_GAP_MULTIPLIER = 1.5  # Gap threshold = avg_width * this
Y_GROUP_MULTIPLIER = 0.4  # Row grouping = avg_height * this

# === UI SETTINGS ===
MAX_UPLOAD_SIZE = 10  # MB
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp"]

# === STREAMLIT CONFIG ===
PAGE_TITLE = "🛍️ Retail AI Pipeline"
PAGE_ICON = "🛍️"
LAYOUT = "wide"

# Create necessary directories
for directory in [UPLOADS_DIR, OUTPUTS_DIR, CROPS_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)