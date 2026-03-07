"""
Optimized Facial Recognition System with performance improvements
"""

import threading
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import numpy as np
import face_recognition
from src.base_system import FacialRecognitionSystem


class OptimizedFacialRecognitionSystem(FacialRecognitionSystem):
    """
    Enhanced facial recognition system with performance optimizations
    """
    
    def __init__(self, encodings_path="data/encodings/face_encodings.pkl"):
        super().__init__(encodings_path)
        self.processing_frame = False
        self.frame_queue = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.recognition_history = {}
        self.fps_counter = 0
        self.fps_timestamp = time.time()
        self.current_fps = 0
        self.frame_skip = 2
        self.frame_count = 0
        
    def process_frame_async(self, frame):
        """
        Process frame asynchronously to improve FPS
        
        Args:
            frame: Video frame to process
        """
        if not self.processing_frame:
            self.processing_frame = True
            self.executor.submit(self._process_frame_worker, frame.copy())
    
    def _process_frame_worker(self, frame):
        """
        Worker function for async frame processing
        
        Args:
            frame: Video frame to process
        """
        try:
            # Resize frame for faster processing
            scale_factor = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(
                rgb_small_frame, 
                model="hog"  # 'hog' is faster than 'cnn'
            )
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Scale back locations
            face_locations = [(int(top/scale_factor), int(right/scale_factor), 
                             int(bottom/scale_factor), int(left/scale_factor)) 
                             for (top, right, bottom, left) in face_locations]
            
            self.frame_queue.append((frame, face_locations, face_encodings))
            
            # Keep queue size manageable
            if len(self.frame_queue) > 5:
                self.frame_queue.pop(0)
                
        except Exception as e:
            print(f"Error in frame processing: {e}")
        finally:
            self.processing_frame = False
    
    def get_latest_processed_frame(self):
        """
        Get the most recently processed frame
        
        Returns:
            tuple: (frame, face_locations, face_encodings) or (None, [], [])
        """
        if self.frame_queue:
            return self.frame_queue[-1]
        return None, [], []
    
    def real_time_recognition(self, tolerance=None, camera_id=0):
        """
        Optimized real-time face recognition
        
        Args:
            tolerance (float): Recognition tolerance
            camera_id (int): Camera device ID
        """
        if len(self.known_face_encodings) == 0:
            print("No known faces registered. Please register faces first.")
            return
        
        tolerance = tolerance or self.tolerance
        video_capture = cv2.VideoCapture(camera_id)
        
        # Optimize camera settings
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print("Starting optimized real-time recognition. Press 'q' to quit, 's' to save frame")
        
        self.fps_timestamp = time.time()
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            self.frame_count += 1
            
            # Update FPS counter
            self.fps_counter += 1
            if time.time() - self.fps_timestamp >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_timestamp = time.time()
            
            # Process frame asynchronously (skip frames for performance)
            if self.frame_count % self.frame_skip == 0:
                self.process_frame_async(frame)
            
            # Get latest processed results
            processed_frame, face_locations, face_encodings = self.get_latest_processed_frame()
            
            if processed_frame is not None:
                # Recognize faces
                face_names = []
                for face_encoding in face_encodings:
                    name = self._match_face(face_encoding, tolerance)
                    face_names.append(name)
                
                # Draw results
                frame = self._draw_recognition_results(
                    processed_frame, face_locations, face_names
                )
            
            # Display FPS and info
            cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(face_locations) if face_locations else 0}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Optimized Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_frame(frame)
        
        video_capture.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=False)
    
    def _match_face(self, face_encoding, tolerance):
        """
        Match face encoding against known faces
        
        Args:
            face_encoding: Face encoding to match
            tolerance: Recognition tolerance
            
        Returns:
            str: Name of matched face or "Unknown"
        """
        if len(self.known_face_encodings) == 0:
            return "Unknown"
        
        matches = face_recognition.compare_faces(
            self.known_face_encodings, face_encoding, tolerance
        )
        
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, face_encoding
        )
        
        if len(face_distances) > 0 and True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                
                # Log high-confidence recognitions
                if confidence > 0.7:
                    self.log_recognition(name)
                
                return name
        
        return "Unknown"
    
    def _draw_recognition_results(self, frame, face_locations, face_names):
        """
        Draw recognition results on frame
        
        Args:
            frame: Video frame
            face_locations: List of face locations
            face_names: List of recognized names
            
        Returns:
            Frame with drawings
        """
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label with background
            label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            label_top = bottom - 35 if bottom - 35 > top else top + 20
            
            cv2.rectangle(frame, 
                         (left, label_top), 
                         (left + label_size[0] + 10, bottom), 
                         color, cv2.FILLED)
            
            cv2.putText(frame, name, (left + 5, bottom - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _save_frame(self, frame):
        """Save current frame"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captured_frames/recognition_{timestamp}.jpg"
        import os
        os.makedirs("captured_frames", exist_ok=True)
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")