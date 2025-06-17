"""
Utility functions for the Retail AI Pipeline
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Dict

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def spatial_weight(distance: float, sigma: float = 0.3) -> float:
    """Calculate spatial weight based on distance"""
    return np.exp(-distance**2 / (2 * sigma**2))

def apply_caqe(vectors: List[np.ndarray], positions: List[Tuple[float, float]]) -> List[np.ndarray]:
    """
    Apply Context-Aware Query Enhancement (CAQE) to vectors
    """
    refined = []
    for i, q in enumerate(vectors):
        q_prime = q.copy()
        for j, n in enumerate(vectors):
            if i == j: 
                continue
            
            alpha = cosine_similarity(q, n)
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx**2 + dy**2)
            beta = spatial_weight(dist)
            q_prime += alpha * beta * n
        
        q_prime = q_prime / np.linalg.norm(q_prime)
        refined.append(q_prime)
    
    return refined

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file and return path"""
    try:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        return None

def create_annotated_image(img: np.ndarray, boxes: np.ndarray, labels: List[str], 
                          gaps: List[Tuple[int, int]] = None) -> np.ndarray:
    """Create annotated image with bounding boxes and gaps"""
    # Ensure image is in BGR format for OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        # If image is in RGB, convert to BGR
        if np.mean(img[:, :, 0]) > np.mean(img[:, :, 2]):  # Check if R channel > B channel
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    annotated = img.copy()
    
    # Draw bounding boxes and labels
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        # Green boxes for products
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with background
        label = labels[i] if i < len(labels) else f"Product {i+1}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated, (x1, y1-25), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Draw gaps as red circles
    if gaps:
        for gx, gy in gaps:
            cv2.circle(annotated, (gx, gy), radius=15, color=(0, 0, 255), thickness=-1)
            cv2.putText(annotated, "GAP", (gx-15, gy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Convert back to RGB for display
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def detect_gaps(boxes: np.ndarray, labels: List[str], x_gap_multiplier: float = 1.5,
                y_group_multiplier: float = 0.4) -> List[Tuple[int, int]]:
    """
    Enhanced gap detection that accurately handles multiple rows.
    
    Algorithm:
    1. Group products into rows based on y-coordinates
    2. For each row:
        - Sort products by x-coordinate
        - Calculate gaps between adjacent products
        - Mark gaps that are significantly larger than average product width
    
    Args:
        boxes: Array of [x1, y1, x2, y2] coordinates
        labels: List of product labels
        x_gap_multiplier: Threshold multiplier for horizontal gaps
        y_group_multiplier: Threshold for grouping products into rows
    
    Returns:
        List of (x, y) coordinates where gaps are detected
    """
    if len(boxes) < 2:
        return []

    # Calculate centers and dimensions
    centers = []
    widths = []
    heights = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        centers.append((cx, cy))
        widths.append(w)
        heights.append(h)
    
    centers = np.array(centers)
    avg_height = np.mean(heights)
    y_threshold = avg_height * y_group_multiplier

    # Group products into rows
    rows = []
    used_indices = set()

    # Sort centers by y-coordinate to process from top to bottom
    y_sorted_indices = np.argsort(centers[:, 1])
    
    for idx in y_sorted_indices:
        if idx in used_indices:
            continue
            
        current_row = [idx]
        used_indices.add(idx)
        current_y = centers[idx][1]

        # Find all products in the same row
        for j in y_sorted_indices:
            if j not in used_indices:
                if abs(centers[j][1] - current_y) < y_threshold:
                    current_row.append(j)
                    used_indices.add(j)
        
        if len(current_row) > 0:
            rows.append(current_row)

    # Process each row for gaps
    gap_centers = []
    for row_indices in rows:
        if len(row_indices) < 2:
            continue

        # Sort products in row by x-coordinate
        row_sorted = sorted(row_indices, key=lambda i: centers[i][0])
        
        # Calculate average product width for this row
        row_widths = [widths[i] for i in row_indices]
        avg_width = np.mean(row_widths)
        min_gap_size = avg_width * x_gap_multiplier

        # Check for gaps between adjacent products
        for i in range(len(row_sorted) - 1):
            current_idx = row_sorted[i]
            next_idx = row_sorted[i + 1]
            
            # Get box coordinates
            current_box = boxes[current_idx]
            next_box = boxes[next_idx]
            
            # Calculate gap size (distance between right edge of current and left edge of next)
            gap_size = next_box[0] - current_box[2]
            
            if gap_size > min_gap_size:
                # Calculate gap center coordinates
                gap_x = (current_box[2] + next_box[0]) / 2
                gap_y = (centers[current_idx][1] + centers[next_idx][1]) / 2
                
                # Add gap center to results
                gap_centers.append((int(gap_x), int(gap_y)))

    return gap_centers

def create_analysis_charts(results: Dict) -> List[go.Figure]:
    """Create analysis charts for SKU distribution"""
    charts = []
    
    # Create pie chart
    pie_fig = go.Figure(data=[go.Pie(
        labels=list(results["sku_counts"].keys()),
        values=list(results["sku_counts"].values()),
        marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    )])
    pie_fig.update_layout(title="Product Distribution")
    charts.append(pie_fig)
    
    # Create bar chart
    bar_fig = go.Figure(data=[go.Bar(
        x=list(results["sku_counts"].keys()),
        y=list(results["sku_counts"].values()),
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    )])
    bar_fig.update_layout(title="Brand Count")
    charts.append(bar_fig)
    
    return charts

def display_progress_bar(progress: float, text: str):
    """Display a progress bar with text"""
    progress_bar = st.progress(progress)
    status_text = st.empty()
    status_text.text(text)
    return progress_bar, status_text

def load_reference_skus(reference_dir: str) -> List[str]:
    """Load available SKU names from reference directory"""
    if not os.path.exists(reference_dir):
        return []
    
    skus = []
    for item in os.listdir(reference_dir):
        item_path = os.path.join(reference_dir, item)
        if os.path.isdir(item_path):
            skus.append(item)
    
    return sorted(skus)

def format_results_summary(total_products: int, unique_brands: int, gaps_found: int) -> str:
    """Format results summary text"""
    return f"""
    📊 **Analysis Results:**
    - **Total Products Detected:** {total_products}
    - **Unique Brands Found:** {unique_brands}
    - **Shelf Gaps Detected:** {gaps_found}
    """