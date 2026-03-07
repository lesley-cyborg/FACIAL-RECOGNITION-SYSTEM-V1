#!/usr/bin/env python3
"""
Training script for facial recognition model
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base_system import FacialRecognitionSystem
from src.batch_processor import BatchFacialRecognition


def main():
    parser = argparse.ArgumentParser(description='Train facial recognition model')
    parser.add_argument('--data-dir', required=True, help='Directory with training images')
    parser.add_argument('--mapping', help='CSV file with name mappings')
    parser.add_argument('--output', default='data/encodings/face_encodings.pkl', 
                       help='Output path for encodings')
    parser.add_argument('--recursive', action='store_true', 
                       help='Search subdirectories recursively')
    
    args = parser.parse_args()
    
    # Initialize system
    system = FacialRecognitionSystem(args.output)
    batch_processor = BatchFacialRecognition(system)
    
    # Batch register faces
    print(f"Training from directory: {args.data_dir}")
    registered, failed = batch_processor.batch_register_from_folder(
        args.data_dir,
        args.mapping,
        recursive=args.recursive
    )
    
    print(f"\nTraining complete!")
    print(f"Successfully registered: {len(registered)}")
    print(f"Failed: {len(failed)}")
    
    # Export encodings
    csv_path = args.output.replace('.pkl', '.csv')
    batch_processor.export_encodings_to_csv(csv_path)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())