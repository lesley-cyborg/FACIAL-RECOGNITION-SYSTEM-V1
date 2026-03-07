"""
Integration tests for facial recognition system
"""

import unittest
import os
import tempfile
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.base_system import FacialRecognitionSystem
from src.optimized_system import OptimizedFacialRecognitionSystem
from src.advanced_recognition import AdvancedRecognitionSystem
from src.database_system import DatabaseFacialRecognition
from src.secure_system import SecureFacialRecognition
from src.batch_processor import BatchFacialRecognition
from src.config_manager import ConfigManager


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.encodings_path = os.path.join(self.temp_dir, 'encodings.pkl')
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        
        # Create test images
        self.create_test_images()
        
    def create_test_images(self):
        """Create test images with simulated faces"""
        self.images_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Create simple face-like images
        for i in range(3):
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            # Draw face oval
            cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (255, 255, 255), -1)
            # Draw eyes
            cv2.circle(img, (70, 80), 10, (0, 0, 0), -1)
            cv2.circle(img, (130, 80), 10, (0, 0, 0), -1)
            # Draw mouth
            cv2.ellipse(img, (100, 120), (30, 15), 0, 0, 180, (0, 0, 0), 2)
            
            cv2.imwrite(os.path.join(self.images_dir, f'face_{i}.jpg'), img)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_full_registration_pipeline(self, mock_encodings, mock_locations):
        """Test complete registration pipeline"""
        # Mock face detection
        mock_locations.return_value = [(50, 150, 150, 50)]
        mock_encodings.return_value = [np.random.rand(128)]
        
        # Test different system types
        systems = [
            ('Basic', FacialRecognitionSystem(self.encodings_path)),
            ('Optimized', OptimizedFacialRecognitionSystem(self.encodings_path)),
            ('Advanced', AdvancedRecognitionSystem(self.encodings_path)),
            ('Database', DatabaseFacialRecognition(self.db_path, self.encodings_path)),
            ('Secure', SecureFacialRecognition(None, self.db_path, self.encodings_path))
        ]
        
        for name, system in systems:
            with self.subTest(system=name):
                # Register face
                image_path = os.path.join(self.images_dir, 'face_0.jpg')
                result = system.register_face_from_image(image_path, f"Test_{name}")
                
                self.assertTrue(result)
                self.assertEqual(len(system.known_face_names), 1)
                self.assertEqual(system.known_face_names[0], f"Test_{name}")
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.compare_faces')
    @patch('face_recognition.face_distance')
    def test_recognition_pipeline(self, mock_distance, mock_compare, 
                                  mock_encodings, mock_locations):
        """Test recognition pipeline"""
        # Setup mocks
        mock_locations.return_value = [(50, 150, 150, 50)]
        mock_encodings.return_value = [np.random.rand(128)]
        mock_compare.return_value = [True]
        mock_distance.return_value = [0.1]
        
        # Create system with known face
        system = FacialRecognitionSystem(self.encodings_path)
        test_encoding = np.random.rand(128)
        system.known_face_encodings = [test_encoding]
        system.known_face_names = ["Test User"]
        
        # Test recognition
        image_path = os.path.join(self.images_dir, 'face_0.jpg')
        
        # Should not raise exceptions
        try:
            system.recognize_faces_from_image(image_path)
            success = True
        except:
            success = False
        
        self.assertTrue(success)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        system = FacialRecognitionSystem(self.encodings_path)
        batch_processor = BatchFacialRecognition(system)
        
        # Test batch registration (with mocks)
        with patch('face_recognition.face_locations') as mock_locations, \
             patch('face_recognition.face_encodings') as mock_encodings:
            
            mock_locations.return_value = [(50, 150, 150, 50)]
            mock_encodings.return_value = [np.random.rand(128)]
            
            registered, failed = batch_processor.batch_register_from_folder(
                self.images_dir
            )
            
            self.assertEqual(len(registered), 3)
            self.assertEqual(len(failed), 0)
    
    def test_config_manager(self):
        """Test configuration management"""
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create test config
        with open(config_path, 'w') as f:
            f.write("""
recognition:
  tolerance: 0.5
  model: cnn
performance:
  target_fps: 60
            """)
        
        # Load config
        config_manager = ConfigManager(config_path)
        
        # Test config values
        self.assertEqual(config_manager.get('recognition', 'tolerance'), 0.5)
        self.assertEqual(config_manager.get('recognition', 'model'), 'cnn')
        self.assertEqual(config_manager.get('performance', 'target_fps'), 60)
        
        # Test default values
        self.assertEqual(
            config_manager.get('storage', 'encodings_path', default='default'),
            'data/encodings/face_encodings.pkl'
        )
    
    def test_system_persistence(self):
        """Test system state persistence"""
        # Create system with data
        system1 = FacialRecognitionSystem(self.encodings_path)
        system1.known_face_names = ["User1", "User2"]
        system1.known_face_encodings = [np.random.rand(128), np.random.rand(128)]
        system1.save_encodings()
        
        # Create new system and load data
        system2 = FacialRecognitionSystem(self.encodings_path)
        
        self.assertEqual(len(system2.known_face_names), 2)
        self.assertEqual(system2.known_face_names, ["User1", "User2"])
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()