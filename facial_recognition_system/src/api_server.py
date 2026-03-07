"""
REST API for Facial Recognition System
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import json
import os
from datetime import datetime
import tempfile
import face_recognition


class FacialRecognitionAPI:
    """
    Flask REST API for facial recognition system
    """
    
    def __init__(self, recognition_system, secret_key=None):
        """
        Initialize API server
        
        Args:
            recognition_system: Facial recognition system instance
            secret_key: Secret key for JWT
        """
        self.app = Flask(__name__)
        self.system = recognition_system
        self.secret_key = secret_key or os.environ.get('SECRET_KEY', 'dev-secret-key')
        
        # Enable CORS
        CORS(self.app)
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/')
        def index():
            """Serve web interface"""
            try:
                return render_template('index.html')
            except:
                # Fallback if template not found
                return jsonify({
                    'message': 'Facial Recognition API is running',
                    'endpoints': {
                        'health': '/api/health',
                        'recognize': '/api/recognize',
                        'register': '/api/register',
                        'faces': '/api/faces',
                        'stats': '/api/stats',
                        'history': '/api/history'
                    }
                })
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'system': 'Facial Recognition API',
                'version': '2.0.0'
            })
        
        @self.app.route('/api/recognize', methods=['POST'])
        def recognize():
            """
            Recognize faces in image
            
            Expected JSON:
            {
                "image": "base64_encoded_image",
                "tolerance": 0.6 (optional)
            }
            """
            try:
                data = request.json
                
                if 'image' not in data:
                    return jsonify({'error': 'No image provided'}), 400
                
                # Decode base64 image
                image_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(BytesIO(image_data))
                image = np.array(image)
                
                # Convert RGB to BGR if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Perform recognition
                tolerance = data.get('tolerance', getattr(self.system, 'tolerance', 0.6))
                
                # Find faces
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                results = []
                for face_location, face_encoding in zip(face_locations, face_encodings):
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.system.known_face_encodings, 
                        face_encoding, 
                        tolerance
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
                    
                    if True in matches and len(self.system.known_face_encodings) > 0:
                        face_distances = face_recognition.face_distance(
                            self.system.known_face_encodings, 
                            face_encoding
                        )
                        
                        # Get all matches above threshold
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
                    
                    results.append(face_result)
                
                return jsonify({
                    'success': True,
                    'faces_detected': len(results),
                    'results': results
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/register', methods=['POST'])
        def register():
            """
            Register a new face
            
            Expected JSON:
            {
                "image": "base64_encoded_image",
                "name": "Person Name",
                "email": "email@example.com" (optional),
                "department": "Department" (optional)
            }
            """
            try:
                data = request.json
                
                if 'image' not in data or 'name' not in data:
                    return jsonify({'error': 'Missing image or name'}), 400
                
                # Decode base64 image
                image_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(BytesIO(image_data))
                image = np.array(image)
                
                # Convert RGB to BGR
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Save temporary image
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"temp_face_{datetime.now().timestamp()}.jpg")
                cv2.imwrite(temp_path, image)
                
                # Register face
                if hasattr(self.system, 'register_person_with_details'):
                    success = self.system.register_person_with_details(
                        temp_path, 
                        data['name'],
                        email=data.get('email'),
                        department=data.get('department')
                    )
                else:
                    success = self.system.register_face_from_image(
                        temp_path, 
                        data['name']
                    )
                
                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                if success:
                    return jsonify({
                        'success': True,
                        'message': f"Registered {data['name']} successfully"
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No face detected in image'
                    }), 400
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/faces', methods=['GET'])
        def list_faces():
            """List all registered faces"""
            try:
                faces = []
                for i, name in enumerate(self.system.known_face_names):
                    face_info = {
                        'id': i + 1,
                        'name': name
                    }
                    
                    # Get additional info from database if available
                    if hasattr(self.system, 'get_person_info'):
                        db_info = self.system.get_person_info(name=name)
                        if db_info:
                            face_info.update({
                                'email': db_info.get('email'),
                                'department': db_info.get('department'),
                                'registration_date': db_info.get('registration_date'),
                                'access_level': db_info.get('access_level')
                            })
                    
                    faces.append(face_info)
                
                return jsonify({
                    'success': True,
                    'total_faces': len(faces),
                    'faces': faces
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/faces/<name>', methods=['DELETE'])
        def delete_face(name):
            """Delete a registered face"""
            try:
                if name in self.system.known_face_names:
                    self.system.delete_face(name)
                    return jsonify({
                        'success': True,
                        'message': f"Deleted {name}"
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f"Face {name} not found"
                    }), 404
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get system statistics"""
            try:
                if hasattr(self.system, 'get_statistics'):
                    stats = self.system.get_statistics()
                else:
                    stats = {
                        'total_faces': len(self.system.known_face_names),
                        'faces': self.system.known_face_names
                    }
                
                return jsonify({
                    'success': True,
                    'stats': stats
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/history', methods=['GET'])
        def get_history():
            """Get recognition history"""
            try:
                person_name = request.args.get('person')
                limit = int(request.args.get('limit', 100))
                
                if hasattr(self.system, 'get_recognition_history'):
                    history = self.system.get_recognition_history(person_name, limit)
                else:
                    # Read from log file
                    history = []
                    log_path = "data/logs/recognition_log.txt"
                    if os.path.exists(log_path):
                        with open(log_path, 'r') as f:
                            lines = f.readlines()[-limit:]
                            for line in lines:
                                if ' - Recognized: ' in line:
                                    parts = line.strip().split(' - Recognized: ')
                                    if len(parts) == 2:
                                        timestamp, name = parts
                                        history.append({
                                            'timestamp': timestamp,
                                            'name': name
                                        })
                
                return jsonify({
                    'success': True,
                    'total': len(history),
                    'history': history
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export', methods=['GET'])
        def export_data():
            """Export system data"""
            try:
                format_type = request.args.get('format', 'json')
                
                if hasattr(self.system, 'export_data'):
                    data = self.system.export_data(format_type)
                    return jsonify(data)
                else:
                    return jsonify({
                        'faces': self.system.known_face_names,
                        'count': len(self.system.known_face_names)
                    })
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/capture', methods=['POST'])
        def capture_frame():
            """Capture and save current frame"""
            try:
                data = request.json
                
                if 'image' not in data:
                    return jsonify({'error': 'No image provided'}), 400
                
                # Decode base64 image
                image_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(BytesIO(image_data))
                image = np.array(image)
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_frames/api_capture_{timestamp}.jpg"
                os.makedirs("captured_frames", exist_ok=True)
                
                cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                return jsonify({
                    'success': True,
                    'filename': filename
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """
        Run the API server
        
        Args:
            host: Host to bind to
            port: Port to listen on
            debug: Debug mode
        """
        print(f"Starting API server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)