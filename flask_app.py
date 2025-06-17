from flask import Flask, request, jsonify, send_from_directory
import traceback
import os
import numpy as np
from pipeline import RetailPipeline
import config
import logging
from flask_cors import CORS
from train_embeddings import train_embedding_model
import base64
from PIL import Image
import io
import json
from werkzeug.utils import secure_filename
import uuid
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
pipeline = RetailPipeline()

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('clean_sku_dataset', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables to track training progress
training_status = {
    'is_training': False,
    'progress': 0,
    'current_loss': 0,
    'status_message': '',
    'error': None
}

@app.route('/')
def home():
    """Home endpoint to verify API is running"""
    return jsonify({
        'status': 'online',
        'endpoints': {
            '/': 'This help message',
            '/analyze_shelf': 'POST - Analyze shelf image for specific brands',
            '/train_embeddings': 'POST - Start model training',
            '/training_status': 'GET - Check training progress',
            '/add_product_images': 'POST - Add product images'
        }
    })

@app.route('/train_embeddings', methods=['POST'])
def train_embeddings():
    try:
        if training_status['is_training']:
            return jsonify({
                'error': 'Training is already in progress',
                'progress': training_status['progress'],
                'current_loss': training_status['current_loss'],
                'status_message': training_status['status_message']
            }), 409

        data = request.get_json() or {}
        batch_size = data.get('batch_size', 32)
        num_epochs = data.get('num_epochs', 50)
        learning_rate = data.get('learning_rate', 0.0001)
        margin = data.get('margin', 1.0)

        training_status.update({
            'is_training': True,
            'progress': 0,
            'current_loss': 0,
            'status_message': 'Starting training...',
            'error': None
        })

        try:
            def progress_callback(epoch, total_epochs, batch_loss):
                progress = (epoch) / total_epochs * 100
                training_status.update({
                    'progress': progress,
                    'current_loss': batch_loss,
                    'status_message': f'Training epoch {epoch}/{total_epochs}'
                })
                logger.info(f"Training progress: {progress:.2f}%, Loss: {batch_loss:.4f}")

            # Start training with original dataset
            train_embedding_model(
                config.REFERENCE_DIR,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                margin=margin,
                progress_callback=progress_callback
            )

            training_status.update({
                'is_training': False,
                'progress': 100,
                'status_message': 'Training completed successfully!'
            })

            # Reset the pipeline to load new model
            global pipeline
            pipeline = RetailPipeline()

            return jsonify({
                'status': 'success',
                'message': 'Training completed successfully',
                'final_loss': training_status['current_loss']
            })

        except Exception as e:
            training_status.update({
                'is_training': False,
                'error': str(e),
                'status_message': f'Training failed: {str(e)}'
            })
            raise

    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'error': error_msg,
            'trace': traceback.format_exc()
        }), 500

@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

@app.route('/analyze_shelf', methods=['POST'])
def analyze_shelf():
    try:
        data = request.get_json()
        if not data or 'brands' not in data or 'image_path' not in data:
            return jsonify({'error': 'Missing required fields: brands, image_path'}), 400
        brands = data['brands']
        image_path = data['image_path']
        
        logger.info(f"Analyzing image: {image_path} for brands: {brands}")
        
        if not isinstance(brands, list) or not all(isinstance(b, str) for b in brands):
            return jsonify({'error': 'brands must be a list of strings'}), 400
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found: {image_path}'}), 400
            
        # Preprocess image to match Streamlit implementation
        try:
            image = Image.open(image_path)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save preprocessed image
            preprocessed_path = os.path.join(config.TEMP_DIR, "preprocessed_image.jpg")
            os.makedirs(config.TEMP_DIR, exist_ok=True)
            image.save(preprocessed_path, quality=85, optimize=True)
            
            # Use preprocessed image for analysis
            image_path = preprocessed_path
            logger.info(f"Image preprocessed and saved to {preprocessed_path}")
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Continue with original image if preprocessing fails
            
        # First get initial analysis without brand filter
        initial_analysis = pipeline.analyze_shelf(image_path)
        if 'error' in initial_analysis:
            logger.error(f"Error in initial analysis: {initial_analysis['error']}")
            return jsonify({'error': initial_analysis['error']}), 500
            
        logger.info(f"Total products detected: {len(initial_analysis['all_boxes'])}")
        logger.info(f"All detected labels: {initial_analysis['all_labels']}")
        
        results = {}
        for brand in brands:
            logger.info(f"Processing brand: {brand}")
            analysis = pipeline.analyze_shelf(image_path, brand_filter=brand)
            if 'error' in analysis:
                logger.error(f"Error analyzing brand {brand}: {analysis['error']}")
                results[brand] = {'error': analysis['error']}
                continue
                
            logger.info(f"Brand {brand} - Filtered boxes: {len(analysis['filtered_boxes'])}")
            logger.info(f"Brand {brand} - Filtered labels: {analysis['filtered_labels']}")
            
            # If no products found for this brand
            if len(analysis['filtered_boxes']) == 0:
                results[brand] = {
                    'count': 0,
                    'sequence': 'Yes',  # Default when no products
                    'gap': 0
                }
                continue
            
            # Enhanced sequence detection
            boxes = analysis['filtered_boxes']
            centers = []
            for box in boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append((cx, cy))
            
            # Sort centers by x-coordinate (left to right)
            centers.sort(key=lambda x: x[0])
            
            sequence = 'Yes'
            if len(centers) > 1:
                # Calculate horizontal spacing between adjacent products
                spacings = []
                for i in range(len(centers) - 1):
                    spacing = centers[i + 1][0] - centers[i][0]
                    spacings.append(spacing)
                
                # Calculate average spacing and standard deviation
                avg_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)
                
                # Calculate vertical alignment
                y_coords = [cy for _, cy in centers]
                y_mean = np.mean(y_coords)
                y_std = np.std(y_coords)
                
                # Calculate product widths
                widths = [x2 - x1 for x1, y1, x2, y2 in boxes]
                avg_width = np.mean(widths)
                std_width = np.std(widths)
                
                # Sequence is considered broken if:
                # 1. Vertical alignment is poor (high y_std)
                # 2. Spacing between products is inconsistent (high std_spacing)
                # 3. Product widths are inconsistent (high std_width)
                
                # Calculate thresholds based on average values
                y_threshold = 0.2 * avg_width  # 20% of average width
                spacing_threshold = 0.3 * avg_spacing  # 30% of average spacing
                width_threshold = 0.2 * avg_width  # 20% of average width
                
                if (y_std > y_threshold or 
                    std_spacing > spacing_threshold or 
                    std_width > width_threshold):
                    sequence = 'No'
            
                logger.info(f"Sequence analysis for {brand}:")
                logger.info(f"Vertical std: {y_std:.2f} (threshold: {y_threshold:.2f})")
                logger.info(f"Spacing std: {std_spacing:.2f} (threshold: {spacing_threshold:.2f})")
                logger.info(f"Width std: {std_width:.2f} (threshold: {width_threshold:.2f})")
            
            # Improved gap detection for multiple rows
            gap_count = 0
            if len(centers) > 1:
                # Group products by rows
                rows = {}
                row_height = avg_width * 0.5  # Use half of average width as row height threshold
                
                # Sort boxes by y-coordinate first
                sorted_boxes = sorted(boxes, key=lambda box: box[1])  # Sort by y1
                
                # Group boxes into rows
                current_row = []
                current_y = sorted_boxes[0][1]
                
                for box in sorted_boxes:
                    if abs(box[1] - current_y) <= row_height:
                        current_row.append(box)
            else:
                        if len(current_row) > 1:
                            # Sort boxes in row by x-coordinate
                            current_row.sort(key=lambda box: box[0])
                            # Calculate gaps in this row
                            for i in range(len(current_row) - 1):
                                gap = current_row[i + 1][0] - current_row[i][2]  # x1 of next - x2 of current
                                if gap > avg_width * 1.5:  # Gap is 50% larger than average width
                                    gap_count += 1
                        current_row = [box]
                        current_y = box[1]
                
                        # Check last row
                        if len(current_row) > 1:
                            current_row.sort(key=lambda box: box[0])
                            for i in range(len(current_row) - 1):
                                gap = current_row[i + 1][0] - current_row[i][2]
                                if gap > avg_width * 1.5:
                                    gap_count += 1
                    
            results[brand] = {
                'count': len(analysis['filtered_boxes']),
                'sequence': sequence,
                'gap': gap_count
            }
            logger.info(f"Results for brand {brand}: {results[brand]}")
            
        return jsonify(results)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/add_product_images', methods=['POST'])
def add_product_images():
    try:
        # Try to get product_name from form data first, then from JSON
        product_name = None
        if request.form:
            product_name = request.form.get('product_name')
        if not product_name and request.is_json:
            data = request.get_json()
            product_name = data.get('product_name')

        if not product_name:
            return jsonify({'error': 'Missing product_name'}), 400

        product_name = secure_filename(product_name)
        
        # Create directory structure
        base_dir = 'clean_sku_dataset'
        product_dir = os.path.join(base_dir, product_name)
        
        # Check if product already exists
        is_new_product = not os.path.exists(product_dir)
        
        # Create product directory
        os.makedirs(product_dir, exist_ok=True)
        
        # Create metadata file for new products
        if is_new_product:
            metadata = {
                'product_name': product_name,
                'created_at': str(datetime.datetime.now()),
                'total_images': 0,
                'last_updated': str(datetime.datetime.now())
            }
            with open(os.path.join(product_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)

        saved_images = []
        errors = []

        # Handle files from form data
        if request.files:
            files = request.files.getlist('images')
            if files and files[0].filename != '':
                for idx, file in enumerate(files):
                    try:
                        if file and allowed_file(file.filename):
                            # Generate unique filename
                            filename = secure_filename(file.filename)
                            unique_filename = f"{uuid.uuid4()}_{filename}"
                            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                            
                            # Save file temporarily
                            file.save(temp_path)
                            
                            try:
                                # Open and validate image
                                image = Image.open(temp_path)
                                image.load()  # Verify it's actually an image
                                
                                # Get next available image number
                                existing_images = [f for f in os.listdir(product_dir) if f.startswith('image_') and f.endswith('.jpg')]
                                next_num = len(existing_images) + 1
                                
                                # Save to final location
                                image_path = os.path.join(product_dir, f'image_{next_num}.jpg')
                                image.save(image_path, 'JPEG')
                                saved_images.append(image_path)
                                logger.info(f"Successfully saved image {next_num} to {image_path}")
                                
                            except Exception as e:
                                errors.append(f"Image {idx + 1}: Invalid image format - {str(e)}")
                            finally:
                                # Clean up temporary file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                        else:
                            errors.append(f"Image {idx + 1}: Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
                            
                    except Exception as e:
                        error_msg = f"Image {idx + 1}: Unexpected error - {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        continue

        # Handle images from JSON
        elif request.is_json:
            data = request.get_json()
            images = data.get('images', [])
            
            if not isinstance(images, list):
                return jsonify({'error': 'images must be a list'}), 400

            for idx, img_data in enumerate(images):
                try:
                    if not isinstance(img_data, str):
                        errors.append(f"Image {idx + 1}: Invalid image data format")
                        continue

                    # Handle file path
                    if os.path.isfile(img_data):
                        try:
                            # Open and validate image
                            image = Image.open(img_data)
                            image.load()  # Verify it's actually an image
                            
                            # Get next available image number
                            existing_images = [f for f in os.listdir(product_dir) if f.startswith('image_') and f.endswith('.jpg')]
                            next_num = len(existing_images) + 1
                            
                            # Save to final location
                            image_path = os.path.join(product_dir, f'image_{next_num}.jpg')
                            image.save(image_path, 'JPEG')
                            saved_images.append(image_path)
                            logger.info(f"Successfully saved image {next_num} to {image_path}")
                            continue
                        except Exception as e:
                            errors.append(f"Image {idx + 1}: Invalid image file - {str(e)}")
                            continue

                    # Handle base64 encoded image
                    if img_data.startswith('data:image'):
                        # Remove the data URL prefix
                        img_data = img_data.split(',')[1]
                    elif not img_data.strip():
                        errors.append(f"Image {idx + 1}: Empty image data")
                        continue

                    try:
                        # Decode base64 image
                        image_bytes = base64.b64decode(img_data)
                    except Exception as e:
                        errors.append(f"Image {idx + 1}: Invalid base64 encoding - {str(e)}")
                        continue

                    try:
                        # Open and validate image
                        image = Image.open(io.BytesIO(image_bytes))
                        # Verify it's actually an image by trying to load it
                        image.load()
                    except Exception as e:
                        errors.append(f"Image {idx + 1}: Invalid image format - {str(e)}")
                        continue

                    # Get next available image number
                    existing_images = [f for f in os.listdir(product_dir) if f.startswith('image_') and f.endswith('.jpg')]
                    next_num = len(existing_images) + 1

                    # Save image
                    image_path = os.path.join(product_dir, f'image_{next_num}.jpg')
                    image.save(image_path, 'JPEG')
                    saved_images.append(image_path)
                    logger.info(f"Successfully saved image {next_num} to {image_path}")
                    
                except Exception as e:
                    error_msg = f"Image {idx + 1}: Unexpected error - {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue

        if not saved_images:
            return jsonify({
                'error': 'No images were successfully saved',
                'details': errors
            }), 400

        # Update metadata
        metadata_path = os.path.join(product_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['total_images'] = len(saved_images)
            metadata['last_updated'] = str(datetime.datetime.now())
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

        response = {
            'status': 'success',
            'message': f'Successfully saved {len(saved_images)} images',
            'product_directory': product_dir,
            'saved_images': saved_images,
            'is_new_product': is_new_product,
            'total_images': len(saved_images)
        }
        
        if errors:
            response['warnings'] = errors

        return jsonify(response)

    except Exception as e:
        error_msg = f"Error adding product images: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'error': error_msg,
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8989, debug=False) 