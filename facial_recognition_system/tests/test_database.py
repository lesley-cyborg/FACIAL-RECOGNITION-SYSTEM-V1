"""
Unit tests for database functionality
"""

import unittest
import os
import tempfile
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database_system import DatabaseFacialRecognition


class TestDatabaseFunctionality(unittest.TestCase):
    """Test cases for database integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.encodings_path = os.path.join(self.temp_dir, 'encodings.pkl')
        self.system = DatabaseFacialRecognition(self.db_path, self.encodings_path)
        
        # Create test face encoding
        self.test_encoding = np.random.rand(128)
        self.system.known_face_encodings = [self.test_encoding]
        self.system.known_face_names = ["Test User"]
    
    def test_database_initialization(self):
        """Test database creation and table structure"""
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['persons', 'recognition_log', 'access_control', 'training_data']
        for table in expected_tables:
            self.assertIn(table, tables)
        
        conn.close()
    
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    def test_register_person_with_details(self, mock_encodings, mock_locations):
        """Test registering a person with additional details"""
        mock_locations.return_value = [(50, 150, 150, 50)]
        mock_encodings.return_value = [self.test_encoding]
        
        # Create dummy image file
        image_path = os.path.join(self.temp_dir, 'test_face.jpg')
        with open(image_path, 'w') as f:
            f.write('dummy')
        
        # Register with details
        result = self.system.register_person_with_details(
            image_path,
            "John Doe",
            "john@example.com",
            "Engineering",
            "admin",
            {"employee_id": "12345"}
        )
        
        self.assertTrue(result)
        
        # Verify database entry
        person_info = self.system.get_person_info(name="John Doe")
        self.assertIsNotNone(person_info)
        self.assertEqual(person_info['name'], "John Doe")
        self.assertEqual(person_info['email'], "john@example.com")
        self.assertEqual(person_info['department'], "Engineering")
        self.assertEqual(person_info['access_level'], "admin")
    
    def test_get_person_info(self):
        """Test retrieving person information"""
        # Insert test data directly
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO persons (name, email, department, registration_date, access_level)
            VALUES (?, ?, ?, ?, ?)
        ''', ("Jane Doe", "jane@example.com", "Marketing", datetime.now(), "user"))
        conn.commit()
        conn.close()
        
        # Test retrieval
        info = self.system.get_person_info(name="Jane Doe")
        self.assertIsNotNone(info)
        self.assertEqual(info['name'], "Jane Doe")
        self.assertEqual(info['email'], "jane@example.com")
        
        # Test non-existent person
        info = self.system.get_person_info(name="Nonexistent")
        self.assertIsNone(info)
    
    def test_recognition_logging(self):
        """Test logging recognitions to database"""
        # Insert test person
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO persons (name, registration_date)
            VALUES (?, ?)
        ''', ("Test User", datetime.now()))
        person_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log recognition
        self.system.log_recognition("Test User", 0.95, "cam_1", "Office")
        
        # Verify log entry
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM recognition_log WHERE person_id = ?', (person_id,))
        log = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(log)
        self.assertAlmostEqual(log[3], 0.95)  # confidence
        self.assertEqual(log[4], "cam_1")  # camera_id
        self.assertEqual(log[5], "Office")  # location
    
    def test_access_control_logging(self):
        """Test logging access control attempts"""
        # Insert test person
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO persons (name, registration_date)
            VALUES (?, ?)
        ''', ("Test User", datetime.now()))
        person_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log access attempt
        self.system.log_access_attempt(
            "Test User", 
            True, 
            "Main Door", 
            "reader_1", 
            "Access granted"
        )
        
        # Verify log entry
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM access_control WHERE person_id = ?', (person_id,))
        log = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(log)
        self.assertEqual(log[2], 1)  # access_granted
        self.assertEqual(log[4], "Main Door")  # location
        self.assertEqual(log[5], "reader_1")  # device_id
    
    def test_get_recognition_history(self):
        """Test retrieving recognition history"""
        # Insert test data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert person
        cursor.execute('''
            INSERT INTO persons (name, registration_date)
            VALUES (?, ?)
        ''', ("History User", datetime.now()))
        person_id = cursor.lastrowid
        
        # Insert recognition logs
        for i in range(5):
            timestamp = datetime.now() - timedelta(hours=i)
            cursor.execute('''
                INSERT INTO recognition_log (person_id, timestamp, confidence, camera_id)
                VALUES (?, ?, ?, ?)
            ''', (person_id, timestamp, 0.9, f"cam_{i}"))
        
        conn.commit()
        conn.close()
        
        # Test history retrieval
        history = self.system.get_recognition_history("History User", limit=3)
        self.assertEqual(len(history), 3)
        
        # Test without name filter
        all_history = self.system.get_recognition_history(limit=10)
        self.assertGreaterEqual(len(all_history), 3)
    
    def test_get_statistics(self):
        """Test statistics generation"""
        # Insert test data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert persons
        for name in ["User1", "User2", "User3"]:
            cursor.execute('''
                INSERT INTO persons (name, registration_date, is_active)
                VALUES (?, ?, 1)
            ''', (name, datetime.now()))
        
        # Insert recognition logs
        for i in range(10):
            cursor.execute('''
                INSERT INTO recognition_log (person_id, timestamp, confidence)
                VALUES (1, datetime('now', '-' || ? || ' hours'), 0.9)
            ''', (i,))
        
        conn.commit()
        conn.close()
        
        # Get statistics
        stats = self.system.get_statistics()
        
        self.assertEqual(stats['total_persons'], 3)
        self.assertEqual(stats['total_recognitions'], 10)
        self.assertIn('most_frequent_person', stats)
    
    def test_export_data(self):
        """Test data export functionality"""
        # Insert test data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO persons (name, email, registration_date)
            VALUES (?, ?, ?)
        ''', ("Export User", "export@example.com", datetime.now()))
        conn.commit()
        conn.close()
        
        # Test JSON export
        json_data = self.system.export_data('json')
        self.assertIn('persons', json_data)
        self.assertIn('recognition_history', json_data)
        self.assertIn('statistics', json_data)
        
        # Test CSV export
        csv_data = self.system.export_data('csv')
        self.assertIn('persons', csv_data)
        self.assertIn('history', csv_data)
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()