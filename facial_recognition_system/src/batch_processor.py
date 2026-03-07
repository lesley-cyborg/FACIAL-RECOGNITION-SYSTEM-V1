"""
Batch processing for facial recognition
"""

import os
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from tqdm import tqdm
import pandas as pd


class BatchFacialRecognition:
    """
    Batch processing for face registration and recognition
    """
    
    def __init__(self, recognition_system):
        """
        Initialize batch processor
        
        Args:
            recognition_system: Facial recognition system instance
        """
        self.system = recognition_system
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
    def batch_register_from_folder(self, folder_path, name_mapping_file=None, 
                                   recursive=True):
        """
        Register multiple faces from a folder
        
        Args:
            folder_path: Path to folder containing images
            name_mapping_file: CSV file mapping filenames to names
            recursive: Search subfolders recursively
            
        Returns:
            tuple: (registered, failed) lists
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder not found: {folder_path}")
            return [], []
        
        # Load name mapping if provided
        name_mapping = {}
        if name_mapping_file and os.path.exists(name_mapping_file):
            with open(name_mapping_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        name_mapping[row[0]] = row[1]
        
        # Collect image files
        image_files = []
        if recursive:
            for ext in self.supported_formats:
                image_files.extend(folder.rglob(f"*{ext}"))
        else:
            for ext in self.supported_formats:
                image_files.extend(folder.glob(f"*{ext}"))
        
        print(f"Found {len(image_files)} image files")
        
        registered = []
        failed = []
        
        # Process each image
        for image_path in tqdm(image_files, desc="Registering faces"):
            # Get name from filename or mapping
            filename = image_path.stem
            
            if filename in name_mapping:
                name = name_mapping[filename]
            elif image_path.parent.name != folder.name and recursive:
                # Use folder name as person name
                name = image_path.parent.name
            else:
                name = filename
            
            try:
                if self.system.register_face_from_image(str(image_path), name):
                    registered.append({
                        'path': str(image_path),
                        'name': name,
                        'filename': image_path.name
                    })
                else:
                    failed.append({
                        'path': str(image_path),
                        'name': name,
                        'reason': "No face detected"
                    })
            except Exception as e:
                failed.append({
                    'path': str(image_path),
                    'name': name,
                    'reason': str(e)
                })
        
        # Generate report
        self._generate_batch_report(registered, failed, folder_path)
        
        print(f"\nBatch registration complete:")
        print(f"  Registered: {len(registered)}")
        print(f"  Failed: {len(failed)}")
        
        return registered, failed
    
    def batch_recognize_from_folder(self, folder_path, output_path=None):
        """
        Recognize faces in multiple images
        
        Args:
            folder_path: Path to folder containing images
            output_path: Path to save results
            
        Returns:
            list: Recognition results
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder not found: {folder_path}")
            return []
        
        # Collect image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(folder.glob(f"*{ext}"))
        
        print(f"Found {len(image_files)} image files to process")
        
        results = []
        
        for image_path in tqdm(image_files, desc="Recognizing faces"):
            try:
                # Load image
                image = face_recognition.load_image_file(str(image_path))
                
                # Find faces
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                image_results = {
                    'image': str(image_path),
                    'filename': image_path.name,
                    'faces_detected': len(face_locations),
                    'recognitions': []
                }
                
                for face_location, face_encoding in zip(face_locations, face_encodings):
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.system.known_face_encodings, 
                        face_encoding, 
                        self.system.tolerance
                    )
                    
                    face_result = {
                        'location': {
                            'top': int(face_location[0]),
                            'right': int(face_location[1]),
                            'bottom': int(face_location[2]),
                            'left': int(face_location[3])
                        },
                        'matches': []
                    }
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(
                            self.system.known_face_encodings, 
                            face_encoding
                        )
                        
                        # Get all matches
                        for i, (match, distance) in enumerate(zip(matches, face_distances)):
                            if match:
                                face_result['matches'].append({
                                    'name': self.system.known_face_names[i],
                                    'confidence': float(1 - distance),
                                    'distance': float(distance)
                                })
                        
                        # Sort by confidence
                        face_result['matches'].sort(
                            key=lambda x: x['confidence'], 
                            reverse=True
                        )
                    
                    image_results['recognitions'].append(face_result)
                
                results.append(image_results)
                
            except Exception as e:
                results.append({
                    'image': str(image_path),
                    'filename': image_path.name,
                    'error': str(e)
                })
        
        # Save results
        if output_path:
            self._save_results(results, output_path)
        
        return results
    
    def _generate_batch_report(self, registered, failed, source_folder):
        """
        Generate batch processing report
        
        Args:
            registered: List of successfully registered faces
            failed: List of failed registrations
            source_folder: Source folder path
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'source_folder': str(source_folder),
            'total_processed': len(registered) + len(failed),
            'successful': len(registered),
            'failed': len(failed),
            'success_rate': len(registered) / (len(registered) + len(failed)) * 100 
                          if (len(registered) + len(failed)) > 0 else 0,
            'details': {
                'registered': registered,
                'failed': failed
            }
        }
        
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        
        # Save JSON report
        report_path = f"reports/batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Batch report saved to {report_path}")
        
        # Save CSV summary
        csv_path = report_path.replace('.json', '.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Status', 'Path', 'Name', 'Reason'])
            
            for item in registered:
                writer.writerow(['Success', item['path'], item['name'], ''])
            
            for item in failed:
                writer.writerow(['Failed', item['path'], item['name'], item['reason']])
        
        print(f"CSV summary saved to {csv_path}")
    
    def _save_results(self, results, output_path):
        """
        Save recognition results
        
        Args:
            results: Recognition results
            output_path: Output file path
        """
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save based on file extension
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif output_path.endswith('.csv'):
            # Flatten results for CSV
            flat_results = []
            for result in results:
                if 'error' in result:
                    flat_results.append({
                        'image': result['image'],
                        'filename': result['filename'],
                        'error': result['error']
                    })
                else:
                    for recognition in result['recognitions']:
                        if recognition['matches']:
                            best_match = recognition['matches'][0]
                            flat_results.append({
                                'image': result['image'],
                                'filename': result['filename'],
                                'face_top': recognition['location']['top'],
                                'face_right': recognition['location']['right'],
                                'face_bottom': recognition['location']['bottom'],
                                'face_left': recognition['location']['left'],
                                'recognized_name': best_match['name'],
                                'confidence': best_match['confidence']
                            })
                        else:
                            flat_results.append({
                                'image': result['image'],
                                'filename': result['filename'],
                                'face_top': recognition['location']['top'],
                                'face_right': recognition['location']['right'],
                                'face_bottom': recognition['location']['bottom'],
                                'face_left': recognition['location']['left'],
                                'recognized_name': 'Unknown',
                                'confidence': 0
                            })
            
            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)
        
        print(f"Results saved to {output_path}")
    
    def export_encodings_to_csv(self, output_path):
        """
        Export face encodings to CSV
        
        Args:
            output_path: Output CSV file path
        """
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['name'] + [f'encoding_{i}' for i in range(128)]
            writer.writerow(header)
            
            # Write data
            for name, encoding in zip(
                self.system.known_face_names,
                self.system.known_face_encodings
            ):
                writer.writerow([name] + encoding.tolist())
        
        print(f"Encodings exported to {output_path}")
    
    def import_encodings_from_csv(self, input_path):
        """
        Import face encodings from CSV
        
        Args:
            input_path: Input CSV file path
            
        Returns:
            int: Number of encodings imported
        """
        imported = 0
        
        with open(input_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header
            
            for row in reader:
                name = row[0]
                encoding = np.array([float(x) for x in row[1:]], dtype=np.float64)
                
                self.system.known_face_names.append(name)
                self.system.known_face_encodings.append(encoding)
                imported += 1
        
        # Save to system
        self.system.save_encodings()
        
        print(f"Imported {imported} face encodings")
        return imported