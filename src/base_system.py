"""
Base Facial Recognition System
"""

import face_recognition
import cv2
import numpy as np
import os
import pickle
from datetime import datetime


class FacialRecognitionSystem:
    """
    Base class for facial recognition functionality
    """
    
    def __init__(self, encodings_path="data/encodings/face_encodings.pkl"):
        """
        Initialize the facial recognition system
        
        Args:
            encodings_path (str): Path to save/load face encodings
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.encodings_path = encodings_path
        self.tolerance = 0.6
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(encodings_path), exist_ok=True)
        
        # Load existing encodings if available
        self.load_encodings()
    
    def load_encodings(self):
        """Load previously saved face encodings"""
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} face encodings")
            except Exception as e:
                print(f"Error loading encodings: {e}")
    
    def save_encodings(self):
        """Save face encodings to file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(self.known_face_names)} face encodings")
        except Exception as e:
            print(f"Error saving encodings: {e}")
    
    def register_face_from_image(self, image_path, person_name):
        """
        Register a new face from an image file
        
        Args:
            image_path (str): Path to image file
            person_name (str): Name of the person
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) == 0:
                print(f"No face found in {image_path}")
                return False
            elif len(face_encodings) > 1:
                print(f"Multiple faces found in {image_path}. Using the first one.")
            
            # Add to known faces
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(person_name)
            
            # Save encodings
            self.save_encodings()
            print(f"Registered {person_name} successfully")
            return True
            
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def register_face_from_camera(self, person_name):
        """
        Register a new face using webcam
        
        Args:
            person_name (str): Name of the person
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        video_capture = cv2.VideoCapture(0)
        
        print(f"Press SPACE to capture face for {person_name}, or ESC to cancel")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Display the frame
            cv2.putText(frame, f"Registering: {person_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE key
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find face locations and encodings
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(person_name)
                    self.save_encodings()
                    print(f"Registered {person_name} successfully")
                    break
                else:
                    print("No face detected. Try again.")
            
            elif key == 27:  # ESC key
                print("Registration cancelled")
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        return len(self.known_face_names) > 0
    
    def recognize_faces_from_image(self, image_path, tolerance=None):
        """
        Recognize faces in an image file
        
        Args:
            image_path (str): Path to image file
            tolerance (float): Recognition tolerance (optional)
        """
        if len(self.known_face_encodings) == 0:
            print("No known faces registered. Please register faces first.")
            return
        
        tolerance = tolerance or self.tolerance
        
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find all face locations and encodings in the image
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Convert to OpenCV format for display
            image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance
                )
                name = "Unknown"
                
                # Calculate face distances
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        print(f"Recognized: {name} (confidence: {confidence:.2f})")
                        
                        # Log recognition
                        self.log_recognition(name)
                
                # Draw rectangle around face
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(image_cv2, (left, top), (right, bottom), color, 2)
                
                # Draw label with name
                label_top = bottom - 35 if bottom - 35 > top else top + 20
                cv2.rectangle(image_cv2, (left, label_top), (right, bottom), color, cv2.FILLED)
                cv2.putText(image_cv2, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Display the result
            cv2.imshow('Face Recognition Result', image_cv2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error recognizing faces: {e}")
    
    def real_time_recognition(self, tolerance=None, camera_id=0):
        """
        Real-time face recognition using webcam
        
        Args:
            tolerance (float): Recognition tolerance (optional)
            camera_id (int): Camera device ID
        """
        if len(self.known_face_encodings) == 0:
            print("No known faces registered. Please register faces first.")
            return
        
        tolerance = tolerance or self.tolerance
        video_capture = cv2.VideoCapture(camera_id)
        
        # Set camera properties
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting real-time recognition. Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        process_every_n_frames = 2  # Process every 2nd frame for better performance
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % process_every_n_frames == 0:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find face locations and encodings
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding, 
                        tolerance
                    )
                    name = "Unknown"
                    
                    # Calculate face distances
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                    
                    face_names.append(name)
                
                # Scale back face locations
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw rectangle
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), 
                                 color, cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Display info
            cv2.putText(frame, f"Faces detected: {len(face_locations) if 'face_locations' in locals() else 0}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Real-time Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_frames/recognition_{timestamp}.jpg"
                os.makedirs("captured_frames", exist_ok=True)
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
        
        video_capture.release()
        cv2.destroyAllWindows()
    
    def log_recognition(self, name):
        """
        Log face recognition events
        
        Args:
            name (str): Recognized person name
        """
        try:
            os.makedirs("data/logs", exist_ok=True)
            with open("data/logs/recognition_log.txt", "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - Recognized: {name}\n")
        except Exception as e:
            print(f"Error logging recognition: {e}")
    
    def list_registered_faces(self):
        """List all registered faces"""
        print("\nRegistered Faces:")
        print("-" * 30)
        for i, name in enumerate(self.known_face_names, 1):
            print(f"{i}. {name}")
        print(f"Total: {len(self.known_face_names)} faces")
    
    def delete_face(self, name_or_index):
        """
        Delete a registered face by name or index
        
        Args:
            name_or_index (str or int): Name or index of face to delete
        """
        if isinstance(name_or_index, int):
            index = name_or_index - 1
            if 0 <= index < len(self.known_face_names):
                name = self.known_face_names.pop(index)
                self.known_face_encodings.pop(index)
                self.save_encodings()
                print(f"Deleted {name}")
            else:
                print("Invalid index")
        else:
            if name_or_index in self.known_face_names:
                indices = [i for i, name in enumerate(self.known_face_names) 
                          if name == name_or_index]
                for i in sorted(indices, reverse=True):
                    self.known_face_names.pop(i)
                    self.known_face_encodings.pop(i)
                self.save_encodings()
                print(f"Deleted {len(indices)} entries for {name_or_index}")
            else:
                print(f"{name_or_index} not found")
    
    def get_face_count(self):
        """Get number of registered faces"""
        return len(self.known_face_names)
    
    def clear_all_faces(self):
        """Clear all registered faces"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.save_encodings()
        print("All faces cleared")