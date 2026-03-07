"""
Database-integrated Facial Recognition System
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from src.advanced_recognition import AdvancedRecognitionSystem


class DatabaseFacialRecognition(AdvancedRecognitionSystem):
    """
    Facial recognition system with database integration
    """
    
    def __init__(self, db_path="data/database/face_recognition.db", 
                 encodings_path="data/encodings/face_encodings.pkl"):
        super().__init__(encodings_path)
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                department TEXT,
                registration_date TIMESTAMP,
                face_encoding_hash TEXT UNIQUE,
                metadata TEXT,
                access_level TEXT DEFAULT 'user',
                is_active BOOLEAN DEFAULT 1,
                last_seen TIMESTAMP
            )
        ''')
        
        # Create recognition_log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                timestamp TIMESTAMP,
                confidence REAL,
                camera_id TEXT,
                image_path TEXT,
                location TEXT,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Create access_control table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_control (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                access_granted BOOLEAN,
                timestamp TIMESTAMP,
                location TEXT,
                device_id TEXT,
                reason TEXT,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Create training_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                image_path TEXT,
                encoding BLOB,
                quality_score REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recognition_timestamp ON recognition_log(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_timestamp ON access_control(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_name ON persons(name)')
        
        conn.commit()
        conn.close()
        
        print("Database initialized successfully")
    
    def register_person_with_details(self, image_path, name, email=None, 
                                     department=None, access_level='user',
                                     metadata=None):
        """
        Register person with additional details
        
        Args:
            image_path: Path to face image
            name: Person's name
            email: Email address
            department: Department
            access_level: Access level (admin/user/guest)
            metadata: Additional metadata
            
        Returns:
            bool: True if registration successful
        """
        # First register face
        if self.register_face_from_image(image_path, name):
            # Generate hash of face encoding
            encoding_hash = hashlib.sha256(
                self.known_face_encodings[-1].tobytes()
            ).hexdigest()
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO persons 
                    (name, email, department, registration_date, 
                     face_encoding_hash, metadata, access_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (name, email, department, datetime.now(), 
                      encoding_hash, json.dumps(metadata) if metadata else None,
                      access_level))
                
                conn.commit()
                person_id = cursor.lastrowid
                
                # Store training data
                self._store_training_data(person_id, image_path, 
                                        self.known_face_encodings[-1])
                
                print(f"Registered {name} in database with ID: {person_id}")
                return True
                
            except sqlite3.IntegrityError as e:
                print(f"Database integrity error: {e}")
                return False
            finally:
                conn.close()
        
        return False
    
    def _store_training_data(self, person_id, image_path, encoding):
        """Store training data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO training_data 
                (person_id, image_path, encoding, quality_score, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, image_path, encoding.tobytes(), 1.0, datetime.now()))
            
            conn.commit()
        except Exception as e:
            print(f"Error storing training data: {e}")
        finally:
            conn.close()
    
    def get_person_info(self, name=None, person_id=None):
        """
        Get person information from database
        
        Args:
            name: Person's name
            person_id: Person's ID
            
        Returns:
            dict: Person information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if person_id:
            cursor.execute('SELECT * FROM persons WHERE id = ?', (person_id,))
        elif name:
            cursor.execute('SELECT * FROM persons WHERE name = ?', (name,))
        else:
            cursor.execute('SELECT * FROM persons')
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            person_dict = dict(zip(columns, row))
            if person_dict.get('metadata'):
                person_dict['metadata'] = json.loads(person_dict['metadata'])
            result.append(person_dict)
        
        conn.close()
        
        return result if len(result) > 1 else (result[0] if result else None)
    
    def get_recognition_history(self, person_name=None, limit=100):
        """
        Get recognition history
        
        Args:
            person_name: Filter by person name
            limit: Maximum number of records
            
        Returns:
            list: Recognition history
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if person_name:
            cursor.execute('''
                SELECT p.name, r.timestamp, r.confidence, r.camera_id, r.location
                FROM recognition_log r
                JOIN persons p ON r.person_id = p.id
                WHERE p.name = ?
                ORDER BY r.timestamp DESC
                LIMIT ?
            ''', (person_name, limit))
        else:
            cursor.execute('''
                SELECT p.name, r.timestamp, r.confidence, r.camera_id, r.location
                FROM recognition_log r
                JOIN persons p ON r.person_id = p.id
                ORDER BY r.timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'name': row[0],
                'timestamp': row[1],
                'confidence': row[2],
                'camera_id': row[3],
                'location': row[4]
            }
            for row in rows
        ]
    
    def log_recognition(self, name, confidence=1.0, camera_id="default", location=None):
        """
        Enhanced logging with database storage
        
        Args:
            name: Recognized person name
            confidence: Recognition confidence
            camera_id: Camera identifier
            location: Location of recognition
        """
        # Keep file logging
        super().log_recognition(name)
        
        # Database logging
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get person_id
            cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if result:
                person_id = result[0]
                
                # Update last_seen
                cursor.execute('''
                    UPDATE persons SET last_seen = ? WHERE id = ?
                ''', (datetime.now(), person_id))
                
                # Insert recognition log
                cursor.execute('''
                    INSERT INTO recognition_log 
                    (person_id, timestamp, confidence, camera_id, location)
                    VALUES (?, ?, ?, ?, ?)
                ''', (person_id, datetime.now(), confidence, camera_id, location))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error logging to database: {e}")
        finally:
            conn.close()
    
    def log_access_attempt(self, name, access_granted, location=None, 
                          device_id=None, reason=None):
        """
        Log access control attempt
        
        Args:
            name: Person name
            access_granted: Whether access was granted
            location: Location
            device_id: Device identifier
            reason: Reason for denial
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get person_id
            cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
            result = cursor.fetchone()
            
            if result:
                person_id = result[0]
                
                cursor.execute('''
                    INSERT INTO access_control 
                    (person_id, access_granted, timestamp, location, device_id, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (person_id, access_granted, datetime.now(), 
                      location, device_id, reason))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error logging access attempt: {e}")
        finally:
            conn.close()
    
    def get_statistics(self):
        """
        Get system statistics from database
        
        Returns:
            dict: System statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total registered persons
        cursor.execute('SELECT COUNT(*) FROM persons WHERE is_active = 1')
        stats['total_persons'] = cursor.fetchone()[0]
        
        # Total recognitions
        cursor.execute('SELECT COUNT(*) FROM recognition_log')
        stats['total_recognitions'] = cursor.fetchone()[0]
        
        # Recognitions today
        cursor.execute('''
            SELECT COUNT(*) FROM recognition_log 
            WHERE date(timestamp) = date('now')
        ''')
        stats['recognitions_today'] = cursor.fetchone()[0]
        
        # Access attempts
        cursor.execute('SELECT COUNT(*) FROM access_control')
        stats['total_access_attempts'] = cursor.fetchone()[0]
        
        # Successful vs failed access
        cursor.execute('''
            SELECT access_granted, COUNT(*) 
            FROM access_control 
            GROUP BY access_granted
        ''')
        access_stats = cursor.fetchall()
        stats['access_granted'] = dict(access_stats).get(1, 0)
        stats['access_denied'] = dict(access_stats).get(0, 0)
        
        # Most frequent person
        cursor.execute('''
            SELECT p.name, COUNT(*) as count
            FROM recognition_log r
            JOIN persons p ON r.person_id = p.id
            GROUP BY r.person_id
            ORDER BY count DESC
            LIMIT 1
        ''')
        most_frequent = cursor.fetchone()
        if most_frequent:
            stats['most_frequent_person'] = most_frequent[0]
            stats['most_frequent_count'] = most_frequent[1]
        
        conn.close()
        
        return stats
    
    def export_data(self, format='json'):
        """
        Export database data
        
        Args:
            format: Export format (json or csv)
            
        Returns:
            dict: Exported data
        """
        data = {
            'persons': self.get_person_info(),
            'recognition_history': self.get_recognition_history(limit=1000),
            'statistics': self.get_statistics()
        }
        
        if format == 'json':
            return data
        elif format == 'csv':
            # Implementation for CSV export
            import pandas as pd
            return {
                'persons': pd.DataFrame(data['persons']),
                'history': pd.DataFrame(data['recognition_history'])
            }
        
        return data