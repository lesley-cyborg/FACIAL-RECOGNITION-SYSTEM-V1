"""
Advanced Recognition System with facial features and multi-frame processing
"""

import cv2
import numpy as np
import dlib
from scipy.spatial import distance
from .optimized_system import OptimizedFacialRecognitionSystem


class AdvancedRecognitionSystem(OptimizedFacialRecognitionSystem):
    """
    Advanced facial recognition with landmark detection and feature extraction
    """
    
    def __init__(self, encodings_path="data/encodings/face_encodings.pkl", 
                 predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        super().__init__(encodings_path)
        
        # Initialize dlib's face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.landmark_predictor = dlib.shape_predictor(predictor_path)
            self.landmarks_available = True
        except:
            print("Warning: Landmark predictor not found. Some features will be disabled.")
            self.landmarks_available = False
        
        self.min_face_size = (80, 80)
        self.recognition_history = {}
        
    def extract_face_features(self, face_image, face_location):
        """
        Extract additional facial features for better matching
        
        Args:
            face_image: Image containing the face
            face_location: (top, right, bottom, left) coordinates
            
        Returns:
            dict: Extracted facial features
        """
        features = {}
        
        if not self.landmarks_available:
            return features
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Create dlib rectangle
            rect = dlib.rectangle(face_location[0], face_location[1], 
                                 face_location[2], face_location[3])
            
            # Get landmarks
            landmarks = self.landmark_predictor(gray, rect)
            
            # Extract key facial points
            left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) 
                               for i in range(36, 42)], axis=0)
            right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) 
                                for i in range(42, 48)], axis=0)
            nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
            right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
            
            # Calculate distances and ratios
            eye_distance = distance.euclidean(left_eye, right_eye)
            nose_to_eye_center = distance.euclidean(
                nose, (left_eye + right_eye) / 2
            )
            mouth_width = distance.euclidean(left_mouth, right_mouth)
            
            # Normalize by face size
            face_width = face_location[1] - face_location[0]
            face_height = face_location[2] - face_location[3]
            
            features['eye_distance_ratio'] = eye_distance / face_width
            features['nose_position_ratio'] = nose_to_eye_center / eye_distance
            features['mouth_width_ratio'] = mouth_width / face_width
            features['face_aspect_ratio'] = face_height / face_width
            
            # Calculate eye aspect ratio (for blink detection)
            left_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) 
                           for i in range(36, 42)]
            right_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) 
                            for i in range(42, 48)]
            
            features['left_eye_aspect'] = self._eye_aspect_ratio(left_eye_pts)
            features['right_eye_aspect'] = self._eye_aspect_ratio(right_eye_pts)
            
        except Exception as e:
            print(f"Error extracting face features: {e}")
        
        return features
    
    def _eye_aspect_ratio(self, eye_points):
        """
        Calculate eye aspect ratio for blink detection
        
        Args:
            eye_points: List of eye landmark points
            
        Returns:
            float: Eye aspect ratio
        """
        # Vertical distances
        vertical1 = distance.euclidean(eye_points[1], eye_points[5])
        vertical2 = distance.euclidean(eye_points[2], eye_points[4])
        
        # Horizontal distance
        horizontal = distance.euclidean(eye_points[0], eye_points[3])
        
        # Calculate aspect ratio
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        
        return ear
    
    def enhance_image_quality(self, image):
        """
        Enhance image quality for better recognition
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply mild denoising
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, 5, 7, 21)
        
        # Convert back to BGR if input was color
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def multi_frame_registration(self, frames, person_name, num_frames=5):
        """
        Register face using multiple frames for better accuracy
        
        Args:
            frames: List of video frames
            person_name: Name of the person
            num_frames: Number of frames to use
            
        Returns:
            bool: True if registration successful
        """
        encodings = []
        
        for i, frame in enumerate(frames[:num_frames]):
            # Enhance frame quality
            enhanced = self.enhance_image_quality(frame)
            rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    print(f"Captured frame {i+1}/{num_frames}")
        
        if len(encodings) >= 3:
            # Average multiple encodings for better representation
            avg_encoding = np.mean(encodings, axis=0)
            self.known_face_encodings.append(avg_encoding)
            self.known_face_names.append(person_name)
            self.save_encodings()
            print(f"Registered {person_name} using {len(encodings)} frames")
            return True
        
        print(f"Failed to register: only {len(encodings)} valid frames")
        return False
    
    def register_from_video(self, video_path, person_name, frame_interval=10):
        """
        Register face from video file
        
        Args:
            video_path: Path to video file
            person_name: Name of the person
            frame_interval: Process every nth frame
            
        Returns:
            bool: True if registration successful
        """
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % frame_interval == 0:
                frames.append(frame)
                print(f"Captured frame {len(frames)}", end='\r')
        
        video_capture.release()
        
        if frames:
            return self.multi_frame_registration(frames, person_name)
        
        return False
    
    def detect_blink(self, frame, face_location):
        """
        Detect if person is blinking (for liveness detection)
        
        Args:
            frame: Video frame
            face_location: Face location
            
        Returns:
            bool: True if blinking detected
        """
        if not self.landmarks_available:
            return False
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rect = dlib.rectangle(face_location[0], face_location[1],
                                 face_location[2], face_location[3])
            
            landmarks = self.landmark_predictor(gray, rect)
            
            # Get eye landmarks
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) 
                       for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) 
                        for i in range(42, 48)]
            
            # Calculate eye aspect ratios
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            
            # Average eye aspect ratio
            ear = (left_ear + right_ear) / 2.0
            
            # Blink threshold (typically < 0.2)
            return ear < 0.2
            
        except:
            return False