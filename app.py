"""
Streamlit Frontend for Retail AI Pipeline
"""

import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np
import torch
from datetime import datetime

# Import local modules
from pipeline import RetailPipeline
from utils import (
    save_uploaded_file, 
    create_annotated_image, 
    create_analysis_charts,
    format_results_summary,
    load_reference_skus
)
from train_embeddings import train_embedding_model
from preprocess_data import organize_dataset
import config

# Initialize session state first, before any UI code
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'uploaded_image_path' not in st.session_state:
    st.session_state.uploaded_image_path = None
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'training_loss' not in st.session_state:
    st.session_state.training_loss = 0.0
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 32
if 'num_epochs' not in st.session_state:
    st.session_state.num_epochs = 50
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 0.0001
if 'margin' not in st.session_state:
    st.session_state.margin = 1.0

# Configure Streamlit page
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
    /* Control image sizes */
    .stImage {
        max-width: 600px !important;
        margin: 0 auto;
    }
    /* Control image sizes - override Streamlit's default image behavior */
    stImage {
        margin: 0 auto !important;
        padding: 10px;
    }
    .stImage img {
        max-height: 300px !important;
        width: auto !important;
        object-fit: contain !important;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    /* Make charts more compact */
    .js-plotly-plot {
        max-width: 600px !important;
        margin: 0 auto;
    }
    /* Training section styles */
    .training-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .training-progress {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_pipeline():
    """Load the AI pipeline with caching"""
    if st.session_state.pipeline is None:
        with st.spinner("🔄 Loading AI models... This may take a few moments."):
            st.session_state.pipeline = RetailPipeline()
    return st.session_state.pipeline

def train_model_callback():
    """Callback for training button"""
    if st.session_state.get('is_training', False):
        st.warning("Training is already in progress!")
        return
    
    try:
        st.session_state.is_training = True
        st.session_state.training_progress = 0
        st.session_state.training_loss = 0.0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_metric = st.empty()
        
        # First preprocess the data
        status_text.text("🔄 Preprocessing dataset...")
        processed_dir = config.REFERENCE_DIR + "_processed"
        organize_dataset(config.REFERENCE_DIR, processed_dir)
        
        # Start training with the selected parameters
        status_text.text("🚀 Training embedding model...")
        
        def progress_callback(epoch, total_epochs, batch_loss):
            """Callback to update training progress"""
            progress = (epoch) / total_epochs
            st.session_state.training_progress = progress
            st.session_state.training_loss = batch_loss
            progress_bar.progress(progress)
            loss_metric.metric("Training Loss", f"{batch_loss:.4f}")
            status_text.text(f"🚀 Training epoch {epoch}/{total_epochs}")
        
        train_embedding_model(
            processed_dir,
            batch_size=st.session_state.batch_size,
            num_epochs=st.session_state.num_epochs,
            learning_rate=st.session_state.learning_rate,
            margin=st.session_state.margin,
            progress_callback=progress_callback
        )
        
        status_text.text("✅ Training completed successfully!")
        progress_bar.progress(1.0)
        
        # Reset the pipeline to load new model
        st.session_state.pipeline = None
        load_pipeline()
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
    finally:
        st.session_state.is_training = False

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ Retail AI Pipeline</h1>
        <p>Advanced Product Detection & Shelf Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model status check
        model_exists = os.path.exists(config.YOLO_MODEL_PATH)
        ref_dir_exists = os.path.exists(config.REFERENCE_DIR)
        
        st.subheader("📋 System Status")
        
        if model_exists:
            st.success("✅ YOLO Model: Ready")
        else:
            st.error("❌ YOLO Model: Missing")
            st.info(f"Expected path: `{config.YOLO_MODEL_PATH}`")
        
        if ref_dir_exists:
            available_skus = load_reference_skus(config.REFERENCE_DIR)
            st.success(f"✅ Reference DB: {len(available_skus)} SKUs")
            
            with st.expander("📦 Available SKUs"):
                for sku in available_skus[:10]:
                    st.text(f"• {sku}")
                if len(available_skus) > 10:
                    st.text(f"... and {len(available_skus) - 10} more")
        else:
            st.error("❌ Reference DB: Missing")
            st.info(f"Expected path: `{config.REFERENCE_DIR}/`")
        
        st.divider()
        
        # Analysis settings
        st.subheader("⚙️ Analysis Settings")
        
        confidence = st.slider(
            "Detection Confidence", 
            min_value=0.1, 
            max_value=0.9, 
            value=config.YOLO_CONFIDENCE,
            step=0.05,
            help="Lower values detect more products but may include false positives"
        )
        
        brand_filter = st.selectbox(
            "Brand Filter (Optional)",
            options=["All Brands"] + available_skus if ref_dir_exists else ["All Brands"],
            help="Focus analysis on specific brand"
        )
        
        if brand_filter == "All Brands":
            brand_filter = None
        
        st.divider()
        
        # Training section
        st.subheader("🎯 Model Training")
        with st.expander("Training Settings", expanded=False):
            # Update session state with UI values
            st.session_state.batch_size = st.number_input(
                "Batch Size", 
                min_value=1, 
                max_value=128, 
                value=st.session_state.batch_size
            )
            
            st.session_state.num_epochs = st.number_input(
                "Number of Epochs", 
                min_value=1, 
                max_value=200, 
                value=st.session_state.num_epochs
            )
            
            st.session_state.learning_rate = st.number_input(
                "Learning Rate", 
                min_value=0.00001, 
                max_value=0.1, 
                value=st.session_state.learning_rate,
                format="%.5f"
            )
            
            st.session_state.margin = st.number_input(
                "Triplet Margin", 
                min_value=0.1, 
                max_value=5.0, 
                value=st.session_state.margin
            )
            
            train_button = st.button(
                "🚀 Train Model",
                on_click=train_model_callback,
                disabled=st.session_state.is_training
            )
            
            if st.session_state.is_training:
                st.info("🔄 Training in progress...")
                if st.session_state.training_progress > 0:
                    st.progress(st.session_state.training_progress)
                if st.session_state.training_loss > 0:
                    st.metric("Current Loss", f"{st.session_state.training_loss:.4f}")
        
        st.divider()
        
        # System info
        st.subheader("💻 System Info")
        device_info = "🔥 GPU" if torch.cuda.is_available() else "💾 CPU"
        st.text(f"Device: {device_info}")

    # Main content area
    st.header("📷 Upload an Image for Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and resize image
        image = Image.open(uploaded_file)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        st.image(image, caption="Uploaded Image", width=400)
        
        # Save uploaded file
        temp_path = os.path.join(config.TEMP_DIR, "uploaded_image.jpg")
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        image.save(temp_path, quality=85, optimize=True)
        
        pipeline = load_pipeline()
        with st.spinner("Analyzing image..."):
            results = pipeline.analyze_shelf(temp_path, brand_filter)
            
            if "error" in results:
                st.error(results["error"])
            else:
                annotated_img = create_annotated_image(
                    results["original_image"],
                    results["filtered_boxes"],
                    results["filtered_labels"]
                )
                st.image(annotated_img, caption="Detected Products", width=400)
                
                charts = create_analysis_charts(results)
                for chart in charts:
                    st.plotly_chart(chart)
                
                st.markdown(format_results_summary(
                    results["total_products"],
                    results["unique_brands"],
                    results["gaps_found"]
                ))

if __name__ == "__main__":
    main()