#!/usr/bin/env python3
"""
Evaluation script for facial recognition model
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base_system import FacialRecognitionSystem
from src.batch_processor import BatchFacialRecognition


def main():
    parser = argparse.ArgumentParser(description='Evaluate facial recognition model')
    parser.add_argument('--test-dir', required=True, help='Directory with test images')
    parser.add_argument('--encodings', default='data/encodings/face_encodings.pkl',
                       help='Path to face encodings')
    parser.add_argument('--ground-truth', help='CSV file with ground truth labels')
    
    args = parser.parse_args()
    
    # Load system
    system = FacialRecognitionSystem(args.encodings)
    
    if len(system.known_face_names) == 0:
        print("No face encodings found. Please train the model first.")
        return 1
    
    print(f"Loaded {len(system.known_face_names)} face encodings")
    
    # Process test images
    batch_processor = BatchFacialRecognition(system)
    results = batch_processor.batch_recognize_from_folder(args.test_dir)
    
    # Calculate metrics if ground truth provided
    if args.ground_truth:
        # Load ground truth
        ground_truth = {}
        with open(args.ground_truth, 'r') as f:
            import csv
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    ground_truth[row[0]] = row[1]
        
        # Compare results
        y_true = []
        y_pred = []
        
        for result in results:
            filename = result['filename']
            if filename in ground_truth:
                true_label = ground_truth[filename]
                
                if result.get('recognitions') and len(result['recognitions']) > 0:
                    if result['recognitions'][0].get('matches'):
                        pred_label = result['recognitions'][0]['matches'][0]['name']
                    else:
                        pred_label = 'Unknown'
                else:
                    pred_label = 'Unknown'
                
                y_true.append(true_label)
                y_pred.append(pred_label)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Total samples: {len(y_true)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())