"""
Utility functions for facial recognition system
"""

import os
import logging
import cv2
import numpy as np
from datetime import datetime
import json


def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/encodings',
        'data/database',
        'data/logs',
        'data/exports',
        'captured_frames',
        'reports',
        'models',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directories created successfully")


def setup_logging(config):
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        logging.Logger: Configured logger
    """
    # SAFELY get logging configuration using proper dict methods
    if isinstance(config, dict):
        # Get logging section safely
        logging_config = config.get('logging', {})
        
        # Get log level safely
        log_level_name = logging_config.get('level', 'INFO')
        log_level = getattr(logging, log_level_name.upper(), logging.INFO)
        
        # Get log format safely
        log_format = logging_config.get('format', 'detailed')
        
        # Get log path safely
        storage_config = config.get('storage', {})
        log_path = storage_config.get('log_path', 'data/logs/recognition.log')
    else:
        # Default values if config is not a dict
        log_level = logging.INFO
        log_format = 'detailed'
        log_path = 'data/logs/recognition.log'
    
    # Set format string based on log_format
    if log_format == 'simple':
        format_str = '%(message)s'
    elif log_format == 'json':
        format_str = '{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
    else:  # detailed
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=format_str,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def print_banner():
    """Print system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║     Enhanced Facial Recognition System v2.0              ║
    ║     Real-time face detection and recognition             ║
    ║               By Kagiso Setilo aka Ninja                 ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def draw_fancy_box(img, text, position, font_scale=0.6, thickness=1):
    """
    Draw fancy box with text
    
    Args:
        img: Image to draw on
        text: Text to display
        position: (x, y) position
        font_scale: Font scale
        thickness: Line thickness
        
    Returns:
        Image with box drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    padding = 5
    
    # Draw background
    cv2.rectangle(img, 
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + padding),
                 (0, 0, 0), -1)
    
    # Draw border
    cv2.rectangle(img, 
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + padding),
                 (0, 255, 0), 1)
    
    # Draw text
    cv2.putText(img, text, (x, y - baseline), font, font_scale, (255, 255, 255), thickness)
    
    return img


def resize_and_pad(image, target_size):
    """
    Resize image while maintaining aspect ratio and pad to target size
    
    Args:
        image: Input image
        target_size: (width, height) target size
        
    Returns:
        Resized and padded image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate position
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized image
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def get_timestamp():
    """Get formatted timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data, filename):
    """Save data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filename):
    """Load data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_similarity(encoding1, encoding2):
    """
    Calculate cosine similarity between two face encodings
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        
    Returns:
        float: Similarity score (0-1)
    """
    # Normalize
    encoding1 = encoding1 / np.linalg.norm(encoding1)
    encoding2 = encoding2 / np.linalg.norm(encoding2)
    
    # Calculate cosine similarity
    similarity = np.dot(encoding1, encoding2)
    
    return float(similarity)