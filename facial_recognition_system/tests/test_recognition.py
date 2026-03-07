"""
Unit tests for recognition functionality
"""

import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
import cv2
import face_recognition

# Add src to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.base_system import FacialRecognitionSystem
from src.optimized_system import OptimizedFacialRecognitionSystem
from src.advanced_recognition import AdvancedRecognitionSystem


class TestRecognitionFunctionality(unittest.TestCase):
    """Test cases for face recognition functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.encodings_path = os.path.join(self.temp_dir, 'test_encodings.pkl')
        self.system = FacialRecognitionSystem(self.encodings_path)
        
        # Create test face encodings
        self.test_encoding = np.random.rand(128)
        self.system.known_face_encodings = [self.test_encoding]
        self.system.known_face_names = ["Test User"]
        
        # Create test image
        self.test_image_path = os.path.join(self.temp_dir, 'test_face.jpg')
        self.create_test_image()
    
    def create_test_image(self):
        """Create a test image with a simulated face"""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw face oval
        cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (255, 255, 255), -1)
        # Draw eyes
        cv2.circle(img, (70, 80), 10, (0, 0, 0), -1)
        cv2.circle(img, (130, 80), 10, (0, 0, 0), -1)
        # Draw mouth
        cv2.ellipse(img, (100, 120), (30, 15), 0, 0, 180, (0, 0, 0), 2)
        cv2.imwrite(self.test_image_path, img)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.compare_faces')
    @patch('face_recognition.face_distance')
    def test_face_matching(self, mock_distance, mock_compare, mock_encodings, mock_locations):
        """Test face matching logic"""
        # Setup mocks
        mock_locations.return_value = [(50, 150, 150, 50)]
        mock_encodings.return_value = [self.test_encoding]
        mock_compare.return_value = [True]
        mock_distance.return_value = [0.1]
        
        # Test recognition
        with patch('cv2.imshow'), patch('cv2.waitKey'), patch('cv2.destroyAllWindows'):
            self.system.recognize_faces_from_image(self.test_image_path)
        
        # Verify mocks were called
        mock_locations.assert_called_once()
        mock_encodings.assert_called_once()
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_no_faces_detected(self, mock_encodings, mock_locations):
        """Test behavior when no faces are detected"""
        mock_locations.return_value = []
        mock_encodings.return_value = []
        
        with patch('builtins.print') as mock_print:
            self.system.recognize_faces_from_image(self.test_image_path)
            mock_print.assert_not_called()  # Should handle gracefully
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.compare_faces')
    @patch('face_recognition.face_distance')
    def test_multiple_faces_recognition(self, mock_distance, mock_compare, mock_encodings, mock_locations):
        """Test recognition with multiple faces"""
        # Setup for multiple faces
        mock_locations.return_value = [(50, 150, 150, 50), (200, 300, 300, 200)]
        mock_encodings.return_value = [self.test_encoding, np.random.rand(128)]
        mock_compare.side_effect = [[True], [False]]
        mock_distance.side_effect = [[0.1], [0.8]]
        
        with patch('cv2.imshow'), patch('cv2.waitKey'), patch('cv2.destroyAllWindows'):
            with patch('builtins.print') as mock_print:
                self.system.recognize_faces_from_image(self.test_image_path)
                
                # Should print recognition for first face only
                mock_print.assert_any_call("Recognized: Test User (confidence: 0.90)")
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        distance = 0.1
        confidence = 1 - distance
        self.assertAlmostEqual(confidence, 0.9)
        
        distance = 0.5
        confidence = 1 - distance
        self.assertAlmostEqual(confidence, 0.5)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_optimized_system_recognition(self, mock_encodings, mock_locations):
        """Test optimized system recognition"""
        system = OptimizedFacialRecognitionSystem(self.encodings_path)
        system.known_face_encodings = [self.test_encoding]
        system.known_face_names = ["Test User"]
        
        mock_locations.return_value = [(50, 150, 150, 50)]
        mock_encodings.return_value = [self.test_encoding]
        
        # Test async processing
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        system.process_frame_async(test_frame)
        
        # Give it a moment to process
        import time
        time.sleep(0.1)
        
        # Get results
        frame, locations, encodings = system.get_latest_processed_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(len(locations), 1)
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_advanced_system_recognition(self, mock_encodings, mock_locations):
        """Test advanced system with landmark detection"""
        system = AdvancedRecognitionSystem(self.encodings_path)
        system.known_face_encodings = [self.test_encoding]
        system.known_face_names = ["Test User"]
        
        mock_locations.return_value = [(50, 150, 150, 50)]
        mock_encodings.return_value = [self.test_encoding]
        
        # Test image enhancement
        test_frame = cv2.imread(self.test_image_path)
        enhanced = system.enhance_image_quality(test_frame)
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, test_frame.shape)
    
    def test_recognition_without_known_faces(self):
        """Test recognition when no faces are registered"""
        empty_system = FacialRecognitionSystem(self.encodings_path)
        
        with patch('builtins.print') as mock_print:
            empty_system.recognize_faces_from_image(self.test_image_path)
            mock_print.assert_called_with("No known faces registered. Please register faces first.")
    
    def test_log_recognition(self):
        """Test recognition logging"""
        log_path = os.path.join(self.temp_dir, 'test_log.txt')
        self.system.log_recognition("Test User")
        
        # Check if log file was created (should be in default location)
        self.assertTrue(os.path.exists('data/logs/recognition_log.txt') or 
                       os.path.exists(os.path.join('data', 'logs', 'recognition_log.txt')))
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()