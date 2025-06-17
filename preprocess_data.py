import os
import shutil
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def organize_dataset(source_dir, target_dir='clean_sku_dataset'):
    """
    Organize the dataset by creating a clean structure for training.
    
    Args:
        source_dir (str): Source directory containing the raw images
        target_dir (str): Target directory for organized dataset
    
    Returns:
        dict: Statistics about the organization process
    """
    try:
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        stats = {
            'total_images': 0,
            'processed_images': 0,
            'skipped_images': 0,
            'errors': []
        }
        
        # Walk through the source directory
        for root, dirs, files in os.walk(source_dir):
            # Get the relative path from source directory
            rel_path = os.path.relpath(root, source_dir)
            
            # Skip the root directory itself
            if rel_path == '.':
                continue
                
            # Create corresponding directory in target
            target_path = os.path.join(target_dir, rel_path)
            os.makedirs(target_path, exist_ok=True)
            
            # Process each file
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    stats['total_images'] += 1
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_path, file)
                    
                    try:
                        # Open and validate image
                        with Image.open(source_file) as img:
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Save as JPEG
                            target_file = os.path.splitext(target_file)[0] + '.jpg'
                            img.save(target_file, 'JPEG', quality=95)
                            
                        stats['processed_images'] += 1
                        logger.info(f"Processed: {source_file} -> {target_file}")
                        
                    except Exception as e:
                        stats['skipped_images'] += 1
                        stats['errors'].append(f"Error processing {source_file}: {str(e)}")
                        logger.error(f"Error processing {source_file}: {str(e)}")
                        continue
        
        logger.info(f"Dataset organization completed:")
        logger.info(f"Total images found: {stats['total_images']}")
        logger.info(f"Successfully processed: {stats['processed_images']}")
        logger.info(f"Skipped images: {stats['skipped_images']}")
        
        return stats
        
    except Exception as e:
        error_msg = f"Error organizing dataset: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def validate_image(image_path):
    """
    Validate if an image file is valid and can be opened.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's actually an image
            return True
    except Exception as e:
        logger.error(f"Invalid image {image_path}: {str(e)}")
        return False

def convert_to_jpg(image_path, target_path=None):
    """
    Convert an image to JPEG format.
    
    Args:
        image_path (str): Path to the source image
        target_path (str, optional): Path for the converted image. If None, replaces original.
    
    Returns:
        str: Path to the converted image
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # If no target path specified, replace original
            if target_path is None:
                target_path = os.path.splitext(image_path)[0] + '.jpg'
            
            # Save as JPEG
            img.save(target_path, 'JPEG', quality=95)
            return target_path
            
    except Exception as e:
        logger.error(f"Error converting {image_path}: {str(e)}")
        raise 