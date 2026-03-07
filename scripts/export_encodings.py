#!/usr/bin/env python3
"""
Export face encodings to various formats
"""

import os
import sys
import argparse
import json
import csv
import pickle
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.base_system import FacialRecognitionSystem
from src.batch_processor import BatchFacialRecognition


def export_to_json(encodings, names, output_path):
    """Export encodings to JSON format"""
    data = {
        'names': names,
        'encodings': [enc.tolist() for enc in encodings],
        'count': len(names)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(names)} encodings to {output_path}")


def export_to_csv(encodings, names, output_path):
    """Export encodings to CSV format"""
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['name'] + [f'encoding_{i}' for i in range(128)]
        writer.writerow(header)
        
        # Write data
        for name, encoding in zip(names, encodings):
            writer.writerow([name] + encoding.tolist())
    
    print(f"Exported {len(names)} encodings to {output_path}")


def export_to_numpy(encodings, names, output_path):
    """Export encodings to NumPy format"""
    np.savez(output_path, 
             encodings=np.array(encodings),
             names=np.array(names))
    
    print(f"Exported {len(names)} encodings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export face encodings')
    parser.add_argument('--input', default='data/encodings/face_encodings.pkl',
                       help='Input encodings file (default: data/encodings/face_encodings.pkl)')
    parser.add_argument('--output', required=True,
                       help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv', 'npy', 'all'], 
                       default='json', help='Output format')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Load encodings
    print(f"Loading encodings from {args.input}...")
    try:
        with open(args.input, 'rb') as f:
            data = pickle.load(f)
            encodings = data['encodings']
            names = data['names']
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return 1
    
    print(f"Loaded {len(names)} face encodings")
    
    # Export based on format
    output_base = os.path.splitext(args.output)[0]
    
    if args.format == 'json' or args.format == 'all':
        export_to_json(encodings, names, f"{output_base}.json")
    
    if args.format == 'csv' or args.format == 'all':
        export_to_csv(encodings, names, f"{output_base}.csv")
    
    if args.format == 'npy' or args.format == 'all':
        export_to_numpy(encodings, names, f"{output_base}.npz")
    
    print("\nExport completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())