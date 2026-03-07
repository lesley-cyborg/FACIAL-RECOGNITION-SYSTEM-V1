"""
Unit tests for base facial recognition system
"""

import unittest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.base_system import FacialRecognitionSystem


class TestFacialRecognitionSystem(unittest.TestCase):
    """Test cases for FacialRecognitionSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.encodings_path = os.path.join(self.temp_dir, 'test_encodings.pkl')
        self.system = FacialRecognitionSystem(self.encodings_path)
        
        # Create test image
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_initialization(self):
        """Test system initialization"""
        self.assertEqual(len(self.system.known_face_names), 0)
        self.assertEqual(len(self.system.known_face_encodings), 0)
        self.assertEqual(self.system.encodings_path, self.encodings_path)
        
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_register_face_from_image(self, mock_encodings, mock_locations):
        """Test face registration from image"""
        # Mock face_recognition functions
        mock_locations.return_value = [(100, 200, 300, 400)]
        mock_encodings.return_value = [np.zeros(128)]
        
        # Create temporary image file
        image_path = os.path.join(self.temp_dir, 'test_face.jpg')
        with open(image_path, 'wb') as f:
            f.write(b'dummy image data')
        
        # Test registration
        result = self.system.register_face_from_image(image_path, "Test User")
        
        self.assertTrue(result)
        self.assertEqual(len(self.system.known_face_names), 1)
        self.assertEqual(self.system.known_face_names[0], "Test User")
        self.assertEqual(len(self.system.known_face_encodings), 1)
        
    def test_save_and_load_encodings(self):
        """Test saving and loading face encodings"""
        # Add test data
        test_encoding = np.random.rand(128)
        self.system.known_face_encodings = [test_encoding]
        self.system.known_face_names = ["Test User"]
        
        # Save encodings
        self.system.save_encodings()
        self.assertTrue(os.path.exists(self.encodings_path))
        
        # Create new system and load encodings
        new_system = FacialRecognitionSystem(self.encodings_path)
        
        self.assertEqual(len(new_system.known_face_names), 1)
        self.assertEqual(new_system.known_face_names[0], "Test User")
        np.testing.assert_array_equal(
            new_system.known_face_encodings[0], 
            test_encoding
        )
        
    def test_delete_face_by_name(self):
        """Test deleting face by name"""
        # Add test data
        self.system.known_face_names = ["User1", "User2", "User1"]
        self.system.known_face_encodings = [np.zeros(128)] * 3
        
        # Delete by name
        self.system.delete_face("User1")
        
        self.assertEqual(len(self.system.known_face_names), 1)
        self.assertEqual(self.system.known_face_names[0], "User2")
        
    def test_delete_face_by_index(self):
        """Test deleting face by index"""
        # Add test data
        self.system.known_face_names = ["User1", "User2", "User3"]
        self.system.known_face_encodings = [np.zeros(128)] * 3
        
        # Delete by index
        self.system.delete_face(2)  # Delete User2
        
        self.assertEqual(len(self.system.known_face_names), 2)
        self.assertEqual(self.system.known_face_names, ["User1", "User3"])
        
    def test_clear_all_faces(self):
        """Test clearing all faces"""
        # Add test data
        self.system.known_face_names = ["User1", "User2"]
        self.system.known_face_encodings = [np.zeros(128)] * 2
        
        self.system.clear_all_faces()
        
        self.assertEqual(len(self.system.known_face_names), 0)
        self.assertEqual(len(self.system.known_face_encodings), 0)
        
    def test_get_face_count(self):
        """Test getting face count"""
        self.assertEqual(self.system.get_face_count(), 0)
        
        self.system.known_face_names = ["User1", "User2"]
        self.assertEqual(self.system.get_face_count(), 2)
        
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()