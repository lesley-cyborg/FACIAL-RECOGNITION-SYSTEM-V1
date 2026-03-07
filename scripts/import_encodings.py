#!/usr/bin/env python3
"""
Import face encodings from various formats
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


def import_from_json(input_path):
    """Import encodings from JSON format"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    names = data['names']
    encodings = [np.array(enc) for enc in data['encodings']]
    
    return encodings, names


def import_from_csv(input_path):
    """Import encodings from CSV format"""
    encodings = []
    names = []
    
    with open(input_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        
        for row in reader:
            name = row[0]
            encoding = np.array([float(x) for x in row[1:]], dtype=np.float64)
            
            names.append(name)
            encodings.append(encoding)
    
    return encodings, names


def import_from_numpy(input_path):
    """Import encodings from NumPy format"""
    data = np.load(input_path)
    
    encodings = list(data['encodings'])
    names = list(data['names'])
    
    return encodings, names


def main():
    parser = argparse.ArgumentParser(description='Import face encodings')
    parser.add_argument('--input', required=True,
                       help='Input file path')
    parser.add_argument('--output', default='data/encodings/face_encodings.pkl',
                       help='Output encodings file (default: data/encodings/face_encodings.pkl)')
    parser.add_argument('--format', choices=['json', 'csv', 'npy'], 
                       help='Input format (auto-detected if not specified)')
    parser.add_argument('--merge', action='store_true',
                       help='Merge with existing encodings instead of replacing')
    
    args = parser.parse_args()
    
    # Auto-detect format if not specified
    if not args.format:
        ext = os.path.splitext(args.input)[1].lower()
        if ext == '.json':
            args.format = 'json'
        elif ext == '.csv':
            args.format = 'csv'
        elif ext == '.npy' or ext == '.npz':
            args.format = 'npy'
        else:
            print(f"Error: Could not auto-detect format for {args.input}")
            return 1
    
    # Import encodings
    print(f"Importing encodings from {args.input}...")
    try:
        if args.format == 'json':
            encodings, names = import_from_json(args.input)
        elif args.format == 'csv':
            encodings, names = import_from_csv(args.input)
        elif args.format == 'npy':
            encodings, names = import_from_numpy(args.input)
    except Exception as e:
        print(f"Error importing encodings: {e}")
        return 1
    
    print(f"Imported {len(names)} face encodings")
    
    # Load existing system
    system = FacialRecognitionSystem(args.output)
    
    if args.merge:
        # Merge with existing encodings
        print(f"Merging with {len(system.known_face_names)} existing encodings")
        
        # Check for duplicates (by name)
        existing_names = set(system.known_face_names)
        new_names_count = 0
        
        for name, encoding in zip(names, encodings):
            if name not in existing_names:
                system.known_face_names.append(name)
                system.known_face_encodings.append(encoding)
                new_names_count += 1
        
        print(f"Added {new_names_count} new encodings")
    else:
        # Replace existing encodings
        print("Replacing existing encodings")
        system.known_face_names = names
        system.known_face_encodings = encodings
    
    # Save
    system.save_encodings()
    print(f"Saved {len(system.known_face_names)} encodings to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())