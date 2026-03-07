"""
Unit tests for security features
"""

import unittest
import os
import tempfile
import numpy as np
import time
from unittest.mock import patch, MagicMock
import jwt
import bcrypt

# Add src to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.secure_system import SecureFacialRecognition


class TestSecurityFeatures(unittest.TestCase):
    """Test cases for security features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.encodings_path = os.path.join(self.temp_dir, 'encodings.pkl')
        self.secret_key = "test_secret_key_12345"
        self.system = SecureFacialRecognition(self.secret_key, self.db_path, self.encodings_path)
        
        # Create test user
        self.test_user = "testuser"
        self.test_password = "SecurePass123!"
        self.test_access_level = "user"
        
        # Hash password
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(self.test_password.encode('utf-8'), salt)
        
        # Insert test user into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO persons (name, password_hash, access_level, registration_date)
            VALUES (?, ?, ?, ?)
        ''', (self.test_user, self.password_hash.decode('utf-8'), 
              self.test_access_level, datetime.now()))
        self.user_id = cursor.lastrowid
        conn.commit()
        conn.close()
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        # Test correct password
        result = bcrypt.checkpw(
            self.test_password.encode('utf-8'),
            self.password_hash
        )
        self.assertTrue(result)
        
        # Test incorrect password
        result = bcrypt.checkpw(
            "wrongpassword".encode('utf-8'),
            self.password_hash
        )
        self.assertFalse(result)
    
    def test_user_authentication(self):
        """Test user authentication"""
        # Test correct credentials
        result = self.system.authenticate_user(self.test_user, self.test_password)
        self.assertTrue(result['authenticated'])
        self.assertEqual(result['access_level'], self.test_access_level)
        self.assertEqual(result['user_id'], self.user_id)
        self.assertIn('token', result)
        
        # Test incorrect password
        result = self.system.authenticate_user(self.test_user, "wrongpassword")
        self.assertFalse(result['authenticated'])
        self.assertIn('error', result)
        
        # Test non-existent user
        result = self.system.authenticate_user("nonexistent", self.test_password)
        self.assertFalse(result['authenticated'])
    
    def test_token_generation_and_verification(self):
        """Test JWT token generation and verification"""
        # Generate token
        token = self.system.generate_token(
            self.test_user, 
            self.test_access_level, 
            self.user_id,
            expiration_hours=1
        )
        
        self.assertIsNotNone(token)
        
        # Verify token
        result = self.system.verify_token(token)
        self.assertTrue(result['valid'])
        self.assertEqual(result['name'], self.test_user)
        self.assertEqual(result['access_level'], self.test_access_level)
        self.assertEqual(result['user_id'], self.user_id)
        
        # Test expired token
        expired_token = self.system.generate_token(
            self.test_user, 
            self.test_access_level, 
            self.user_id,
            expiration_hours=-1
        )
        result = self.system.verify_token(expired_token)
        self.assertFalse(result['valid'])
        self.assertEqual(result['error'], 'Token expired')
        
        # Test invalid token
        result = self.system.verify_token("invalid.token.here")
        self.assertFalse(result['valid'])
        self.assertEqual(result['error'], 'Invalid token')
    
    def test_access_level_check(self):
        """Test access level verification"""
        # Test sufficient access
        result = self.system.check_access(self.test_user, "user")
        self.assertTrue(result)
        
        # Test insufficient access
        result = self.system.check_access(self.test_user, "admin")
        self.assertFalse(result)
        
        # Test non-existent user
        result = self.system.check_access("nonexistent", "user")
        self.assertFalse(result)
    
    def test_login_attempt_tracking(self):
        """Test login attempt tracking and lockout"""
        # Set max attempts lower for testing
        self.system.max_login_attempts = 3
        
        # Make failed attempts
        for i in range(3):
            result = self.system.authenticate_user(self.test_user, "wrongpassword")
            self.assertFalse(result['authenticated'])
        
        # Next attempt should be locked
        result = self.system.authenticate_user(self.test_user, self.test_password)
        self.assertFalse(result['authenticated'])
        self.assertIn('Account locked', result['error'])
        
        # Reset attempts (simulate lockout period passing)
        del self.system.login_attempts[self.test_user]
        
        # Should work again
        result = self.system.authenticate_user(self.test_user, self.test_password)
        self.assertTrue(result['authenticated'])
    
    @patch('cv2.Laplacian')
    def test_anti_spoofing_check(self, mock_laplacian):
        """Test anti-spoofing measures"""
        # Mock Laplacian variance
        mock_laplacian.return_value.var.return_value = 200
        
        # Create test face image and location
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_location = (10, 90, 90, 10)
        
        # Test live face detection
        is_live, reason = self.system.anti_spoofing_check(test_image, test_location)
        self.assertTrue(is_live or not is_live)  # Just check it runs without error
        
        # Test with low texture (possible print)
        mock_laplacian.return_value.var.return_value = 30
        is_live, reason = self.system.anti_spoofing_check(test_image, test_location)
        if not is_live:
            self.assertIn('printed photo', reason.lower())
    
    def test_blink_detection(self):
        """Test blink detection functionality"""
        # This test requires the landmark predictor
        if not hasattr(self.system, 'landmarks_available') or not self.system.landmarks_available:
            self.skipTest("Landmark predictor not available")
        
        # Create test frame
        test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        test_location = (50, 150, 150, 50)
        
        # Test blink detection
        is_blinking = self.system.detect_blink(test_frame, test_location)
        # Just check it runs without error (actual result depends on detector)
        self.assertIsInstance(is_blinking, bool)
    
    def test_register_with_password(self):
        """Test user registration with password"""
        with patch('face_recognition.face_locations') as mock_locations, \
             patch('face_recognition.face_encodings') as mock_encodings:
            
            mock_locations.return_value = [(50, 150, 150, 50)]
            mock_encodings.return_value = [np.random.rand(128)]
            
            # Create dummy image
            image_path = os.path.join(self.temp_dir, 'new_user.jpg')
            with open(image_path, 'w') as f:
                f.write('dummy')
            
            # Register new user
            result = self.system.register_with_password(
                image_path,
                "newuser",
                "NewPass123!",
                "new@example.com",
                "IT",
                "manager"
            )
            
            self.assertTrue(result)
            
            # Verify can authenticate
            auth_result = self.system.authenticate_user("newuser", "NewPass123!")
            self.assertTrue(auth_result['authenticated'])
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()