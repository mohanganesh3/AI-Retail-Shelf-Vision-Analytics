"""
Core AI Pipeline for Retail Product Detection and Recognition
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import faiss
from torchvision import models, transforms
from ultralytics import YOLO
from typing import List, Tuple, Dict
from collections import Counter
import logging
import torch.nn as nn

from utils import apply_caqe, detect_gaps
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetailPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = None
        self.resnet_model = None
        self.transform = None
        self.faiss_index = None
        self.sku_labels = []
        self.sku_vectors = []
        
        self._setup_models()
        
    def _setup_models(self):
        """Setup YOLO and ResNet models"""
        # Load YOLO model
        if os.path.exists(config.YOLO_MODEL_PATH):
            logger.info(f"Loading YOLO model from {config.YOLO_MODEL_PATH}")
            self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
        else:
            logger.error(f"YOLO model not found at {config.YOLO_MODEL_PATH}")
            return
        
        # Load ResNet model
        self.resnet_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet_model.fc = nn.Identity()  # Remove final FC layer
        
        # Load trained embeddings if available
        if os.path.exists(config.EMBEDDINGS_PATH):
            try:
                logger.info(f"Loading embeddings from {config.EMBEDDINGS_PATH}")
                checkpoint = torch.load(config.EMBEDDINGS_PATH, map_location=self.device)
                # Load state dict with strict=False to ignore missing keys
                self.resnet_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info("Successfully loaded custom trained embedding model")
            except Exception as e:
                logger.warning(f"Could not load trained embeddings: {str(e)}")
        
        self.resnet_model = self.resnet_model.to(self.device).eval()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225]),
        ])
        
        # Build reference database
        self._build_reference_database()
    
    def _embed_image(self, img_path: str) -> np.ndarray:
        """Extract features from image using ResNet-34"""
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                vec = self.resnet_model(tensor).squeeze().cpu().numpy()
                vec = vec / np.linalg.norm(vec)  # Normalize
            
            return vec
        except Exception as e:
            return None
    
    def _build_reference_database(self):
        """Build FAISS database from reference SKU images"""
        if not os.path.exists(config.REFERENCE_DIR):
            return
        
        self.sku_vectors = []
        self.sku_labels = []
        
        sku_folders = [d for d in os.listdir(config.REFERENCE_DIR) 
                      if os.path.isdir(os.path.join(config.REFERENCE_DIR, d))]
        
        for idx, sku_id in enumerate(sku_folders):
            sku_path = os.path.join(config.REFERENCE_DIR, sku_id)
            raw_vectors = []
            positions = []
            
            img_files = [f for f in os.listdir(sku_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for i, img_file in enumerate(img_files):
                img_path = os.path.join(sku_path, img_file)
                vec = self._embed_image(img_path)
                if vec is not None:
                    raw_vectors.append(vec)
                    positions.append((i, 0))  # Fake spatial position
            
            if raw_vectors:
                # Apply CAQE to SKU group
                refined_vectors = apply_caqe(raw_vectors, positions)
                
                for vec in refined_vectors:
                    self.sku_vectors.append(vec)
                    self.sku_labels.append(sku_id)
        
        # Build FAISS index
        if self.sku_vectors:
            d = self.sku_vectors[0].shape[0]
            self.faiss_index = faiss.IndexFlatL2(d)
            self.faiss_index.add(np.stack(self.sku_vectors))
        else:
            pass
    
    def detect_products(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Detect products in shelf image using YOLO"""
        if not self.yolo_model:
            return None, None
        
        try:
            results = self.yolo_model.predict(
                source=image_path, 
                save=False, 
                conf=config.YOLO_CONFIDENCE
            )
            
            pred = results[0]
            boxes = pred.boxes.xyxy.cpu().numpy()
            img = cv2.imread(image_path)
            
            return img, boxes
        except Exception as e:
            return None, None
    
    def crop_products(self, img: np.ndarray, boxes: np.ndarray) -> List[str]:
        """Crop detected products and save them"""
        crop_paths = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            crop_path = os.path.join(config.CROPS_DIR, f"crop_{i}.jpg")
            cv2.imwrite(crop_path, crop)
            crop_paths.append(crop_path)
        
        return crop_paths
    
    def recognize_skus(self, crop_paths: List[str], boxes: np.ndarray) -> List[str]:
        """Recognize SKUs using ResNet + CAQE + FAISS"""
        if not self.faiss_index:
            return []
        
        # Extract features from crops
        query_vecs = []
        positions = []
        
        for i, path in enumerate(crop_paths):
            vec = self._embed_image(path)
            if vec is not None:
                query_vecs.append(vec)
                
                # Use box center for spatial position
                x1, y1, x2, y2 = boxes[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                positions.append((cx, cy))
        
        if not query_vecs:
            return []
        
        # Apply CAQE to query vectors
        refined_query_vecs = apply_caqe(query_vecs, positions)
        
        # Search in FAISS index with k=1
        D, I = self.faiss_index.search(np.stack(refined_query_vecs), k=1)
        
        # Set similarity threshold (adjust this value based on your needs)
        SIMILARITY_THRESHOLD = 0.7
        
        # Convert distances to similarities (FAISS returns L2 distances)
        similarities = 1 / (1 + D[:, 0])  # Convert distance to similarity score
        
        # Match SKUs with threshold
        matched_skus = []
        for i, (idx, sim) in enumerate(zip(I[:, 0], similarities)):
            if sim >= SIMILARITY_THRESHOLD:
                matched_skus.append(self.sku_labels[idx])
            else:
                matched_skus.append("Unknown")
        
        return matched_skus
    
    def filter_by_brand(self, boxes: np.ndarray, labels: List[str], 
                       brand_filter: str) -> Tuple[np.ndarray, List[str], List[Tuple[float, float]]]:
        """Filter detections by brand name"""
        if not brand_filter:
            return boxes, labels, []
        
        brand_filter = brand_filter.lower()
        filtered_boxes = []
        filtered_labels = []
        filtered_centers = []
        
        logger.info(f"Filtering for brand: {brand_filter}")
        logger.info(f"Available labels: {labels}")
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            label_lower = label.lower()
            # More flexible brand matching
            is_match = (
                brand_filter in label_lower or  # Substring match
                label_lower in brand_filter or  # Label is part of brand name
                any(word in label_lower.split() for word in brand_filter.split())  # Word match
            )
            if is_match:
                logger.info(f"Matched label '{label}' with brand '{brand_filter}'")
                filtered_boxes.append(box)
                filtered_labels.append(label)
                
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                filtered_centers.append((cx, cy))
        
        logger.info(f"Found {len(filtered_boxes)} matches for brand {brand_filter}")
        return np.array(filtered_boxes) if filtered_boxes else np.array([]), filtered_labels, filtered_centers
    
    def analyze_shelf(self, image_path: str, brand_filter: str = None) -> Dict:
        """Complete shelf analysis pipeline"""
        # Step 1: Detect products
        logger.info(f"Detecting products in {image_path}")
        img, boxes = self.detect_products(image_path)
        if img is None or len(boxes) == 0:
            logger.error("No products detected in image")
            return {"error": "No products detected"}
        
        logger.info(f"Detected {len(boxes)} products")
        
        # Step 2: Crop products
        crop_paths = self.crop_products(img, boxes)
        logger.info(f"Created {len(crop_paths)} crops")
        
        # Step 3: Recognize SKUs
        matched_skus = self.recognize_skus(crop_paths, boxes)
        logger.info(f"Recognized SKUs: {matched_skus}")
        
        # Step 4: Filter by brand if specified
        if brand_filter:
            filtered_boxes, filtered_labels, _ = self.filter_by_brand(
                boxes, matched_skus, brand_filter
            )
        else:
            filtered_boxes, filtered_labels = boxes, matched_skus
        
        logger.info(f"After filtering - boxes: {len(filtered_boxes)}, labels: {filtered_labels}")
        
        # Step 5: Detect gaps
        gaps = detect_gaps(
            filtered_boxes, 
            filtered_labels,
            config.X_GAP_MULTIPLIER,
            config.Y_GROUP_MULTIPLIER
        )
        
        # Step 6: Count products
        sku_counts = Counter(filtered_labels)
        
        return {
            "original_image": img,
            "all_boxes": boxes,
            "all_labels": matched_skus,
            "filtered_boxes": filtered_boxes,
            "filtered_labels": filtered_labels,
            "gaps": gaps,
            "sku_counts": dict(sku_counts),
            "total_products": len(filtered_boxes),
            "unique_brands": len(sku_counts),
            "gaps_found": len(gaps)
        }