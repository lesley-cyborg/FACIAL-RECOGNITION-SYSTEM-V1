"""
Secure Facial Recognition System with authentication and anti-spoofing
"""

import bcrypt
import jwt
import secrets
from datetime import datetime, timedelta
import cv2
import numpy as np
from src.database_system import DatabaseFacialRecognition


class SecureFacialRecognition(DatabaseFacialRecognition):
    """
    Secure facial recognition with authentication and anti-spoofing
    """
    
    def __init__(self, secret_key=None, db_path="data/database/face_recognition.db",
                 encodings_path="data/encodings/face_encodings.pkl"):
        super().__init__(db_path, encodings_path)
        self.secret_key = secret_key or secrets.token_hex(32)
        self.access_levels = {
            'admin': 3,
            'manager': 2,
            'user': 1,
            'guest': 0
        }
        self.login_attempts = {}
        self.max_login_attempts = 5
        self.lockout_minutes = 15
        
        # Anti-spoofing parameters
        self.blink_threshold = 0.2
        self.movement_threshold = 10
        self.last_face_position = None
        self.blink_count = 0
        
    def authenticate_user(self, name, password):
        """
        Authenticate user with password
        
        Args:
            name: Username
            password: Password
            
        Returns:
            dict: Authentication result
        """
        # Check for lockout
        if self._is_locked_out(name):
            return {
                'authenticated': False,
                'error': 'Account locked. Too many failed attempts.'
            }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get stored password hash and access level
        cursor.execute('''
            SELECT password_hash, access_level, id FROM persons WHERE name = ?
        ''', (name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            password_hash, access_level, user_id = result
            
            if password_hash and bcrypt.checkpw(password.encode('utf-8'), 
                                               password_hash.encode('utf-8')):
                # Reset login attempts
                self._reset_login_attempts(name)
                
                # Generate token
                token = self.generate_token(name, access_level, user_id)
                
                return {
                    'authenticated': True,
                    'access_level': access_level,
                    'token': token,
                    'user_id': user_id
                }
            else:
                # Record failed attempt
                self._record_failed_attempt(name)
        
        return {
            'authenticated': False,
            'error': 'Invalid credentials'
        }
    
    def register_with_password(self, image_path, name, password, email=None,
                              department=None, access_level='user'):
        """
        Register user with password
        
        Args:
            image_path: Path to face image
            name: Username
            password: Password
            email: Email
            department: Department
            access_level: Access level
            
        Returns:
            bool: True if registration successful
        """
        # Hash password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # Register face first
        if self.register_person_with_details(image_path, name, email, 
                                            department, access_level):
            
            # Update database with password hash
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    UPDATE persons 
                    SET password_hash = ? 
                    WHERE name = ?
                ''', (password_hash.decode('utf-8'), name))
                
                conn.commit()
                print(f"Password set for {name}")
                return True
                
            except Exception as e:
                print(f"Error setting password: {e}")
            finally:
                conn.close()
        
        return False
    
    def generate_token(self, name, access_level, user_id, expiration_hours=24):
        """
        Generate JWT token for authenticated user
        
        Args:
            name: Username
            access_level: Access level
            user_id: User ID
            expiration_hours: Token expiration in hours
            
        Returns:
            str: JWT token
        """
        payload = {
            'name': name,
            'access_level': access_level,
            'user_id': user_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=expiration_hours)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token):
        """
        Verify JWT token
        
        Args:
            token: JWT token
            
        Returns:
            dict: Token verification result
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return {
                'valid': True,
                'name': payload['name'],
                'access_level': payload['access_level'],
                'user_id': payload['user_id']
            }
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
    
    def check_access(self, name, required_level):
        """
        Check if user has required access level
        
        Args:
            name: Username
            required_level: Required access level
            
        Returns:
            bool: True if user has required access
        """
        user_info = self.get_person_info(name=name)
        
        if user_info:
            user_level = self.access_levels.get(user_info.get('access_level', 'guest'), 0)
            required = self.access_levels.get(required_level, 0) if isinstance(required_level, str) else required_level
            return user_level >= required
        
        return False
    
    def _is_locked_out(self, name):
        """Check if user is locked out"""
        if name in self.login_attempts:
            attempts = self.login_attempts[name]
            if attempts['count'] >= self.max_login_attempts:
                lockout_time = attempts['last_attempt'] + timedelta(minutes=self.lockout_minutes)
                if datetime.now() < lockout_time:
                    return True
                else:
                    # Reset after lockout period
                    del self.login_attempts[name]
        return False
    
    def _record_failed_attempt(self, name):
        """Record failed login attempt"""
        if name not in self.login_attempts:
            self.login_attempts[name] = {
                'count': 1,
                'last_attempt': datetime.now()
            }
        else:
            self.login_attempts[name]['count'] += 1
            self.login_attempts[name]['last_attempt'] = datetime.now()
    
    def _reset_login_attempts(self, name):
        """Reset login attempts for user"""
        if name in self.login_attempts:
            del self.login_attempts[name]
    
    def anti_spoofing_check(self, face_image, face_location):
        """
        Perform anti-spoofing checks
        
        Args:
            face_image: Image containing the face
            face_location: Face location coordinates
            
        Returns:
            tuple: (is_live, reason)
        """
        # Check 1: Texture analysis (Laplacian variance)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:  # Too smooth - might be printed photo
            return False, "Possible printed photo detected (low texture)"
        
        # Check 2: Brightness consistency
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        if brightness_std < 20:  # Low contrast - might be screen
            return False, "Possible screen detection (low contrast)"
        
        # Check 3: Blink detection (if landmarks available)
        if self.landmarks_available:
            is_blinking = self.detect_blink(face_image, face_location)
            if is_blinking:
                self.blink_count += 1
                if self.blink_count > 3:
                    return True, "Live face detected (blinks observed)"
        
        # Check 4: Movement detection
        if self.last_face_position is not None:
            movement = np.linalg.norm(
                np.array(face_location[:2]) - np.array(self.last_face_position[:2])
            )
            if movement < self.movement_threshold:
                return False, "Possible static image (no movement)"
        
        self.last_face_position = face_location
        
        # If all checks pass but we haven't seen enough blinks
        if self.blink_count < 2:
            return False, "Insufficient liveness evidence"
        
        return True, "Live face detected"
    
    def secure_real_time_recognition(self, camera_id=0, require_liveness=True):
        """
        Real-time recognition with security features
        
        Args:
            camera_id: Camera device ID
            require_liveness: Whether to require liveness detection
        """
        if len(self.known_face_encodings) == 0:
            print("No known faces registered. Please register faces first.")
            return
        
        video_capture = cv2.VideoCapture(camera_id)
        
        # Reset liveness counter
        self.blink_count = 0
        self.last_face_position = None
        
        print("Starting secure real-time recognition. Press 'q' to quit")
        print("Liveness detection:", "Enabled" if require_liveness else "Disabled")
        
        frame_count = 0
        process_every_n_frames = 2
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % process_every_n_frames == 0:
                # Resize for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                liveness_results = []
                
                for face_location, face_encoding in zip(face_locations, face_encodings):
                    # Scale location back
                    full_location = tuple(x * 4 for x in face_location)
                    
                    # Perform recognition
                    name = self._match_face(face_encoding, self.tolerance)
                    
                    # Perform liveness check if required
                    if require_liveness and name != "Unknown":
                        is_live, reason = self.anti_spoofing_check(frame, full_location)
                        if not is_live:
                            name = f"Spoof? {name}"
                            liveness_results.append((False, reason))
                        else:
                            liveness_results.append((True, "Live"))
                            
                            # Log high-confidence recognition
                            if name != "Unknown":
                                self.log_recognition(name, confidence=0.9, 
                                                   camera_id=f"cam_{camera_id}")
                    
                    face_names.append(name)
                
                # Scale back face locations for display
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Choose color based on recognition and liveness
                    if "Unknown" in name:
                        color = (0, 0, 255)  # Red for unknown
                    elif "Spoof" in name:
                        color = (0, 165, 255)  # Orange for possible spoof
                    else:
                        color = (0, 255, 0)  # Green for genuine
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), 
                                 color, cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Display security status
            status_color = (0, 255, 0) if self.blink_count >= 2 else (0, 0, 255)
            cv2.putText(frame, f"Liveness: {self.blink_count}/2", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow('Secure Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()