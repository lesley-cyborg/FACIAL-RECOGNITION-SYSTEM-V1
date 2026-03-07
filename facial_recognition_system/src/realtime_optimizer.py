"""
Real-time performance optimizer for facial recognition
"""

import time
import numpy as np
import cv2
from collections import deque


class RealTimeOptimizer:
    """
    Optimizes real-time face recognition performance
    """
    
    def __init__(self, target_fps=30, history_size=30):
        """
        Initialize optimizer
        
        Args:
            target_fps: Target frames per second
            history_size: Number of frames to keep in history
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_process_time = time.time()
        self.frame_count = 0
        self.processing_times = deque(maxlen=history_size)
        self.frame_sizes = deque(maxlen=history_size)
        self.quality_scores = deque(maxlen=history_size)
        
        # Adaptive parameters
        self.scale_factor = 0.5
        self.skip_frames = 1
        self.quality_threshold = 0.6
        
    def should_process_frame(self):
        """
        Determine if current frame should be processed
        
        Returns:
            bool: True if frame should be processed
        """
        current_time = time.time()
        if current_time - self.last_process_time >= self.frame_time * self.skip_frames:
            self.last_process_time = current_time
            return True
        return False
    
    def measure_performance(self, processing_time, frame_shape=None):
        """
        Measure and track performance metrics
        
        Args:
            processing_time: Time taken to process frame
            frame_shape: Shape of processed frame
            
        Returns:
            dict: Performance metrics
        """
        self.processing_times.append(processing_time)
        
        if frame_shape:
            self.frame_sizes.append(frame_shape[0] * frame_shape[1])
        
        # Calculate metrics
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        actual_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Calculate stability (inverse of variance)
        if len(self.processing_times) > 5:
            time_variance = np.var(list(self.processing_times)[-5:])
            stability = 1.0 / (1.0 + time_variance)
        else:
            stability = 1.0
        
        metrics = {
            'avg_processing_time': float(avg_time),
            'actual_fps': float(actual_fps),
            'target_fps': self.target_fps,
            'performance_ratio': float(actual_fps / self.target_fps if self.target_fps > 0 else 0),
            'stability': float(stability),
            'skip_frames': self.skip_frames,
            'scale_factor': self.scale_factor
        }
        
        return metrics
    
    def adaptive_resize(self, frame, processing_time=None):
        """
        Dynamically adjust frame size based on processing time
        
        Args:
            frame: Input frame
            processing_time: Time taken to process last frame
            
        Returns:
            numpy.ndarray: Resized frame
        """
        if processing_time is not None:
            # Adjust scale factor based on processing time
            if processing_time > self.frame_time * 1.5:
                # Too slow - reduce size
                self.scale_factor = max(0.25, self.scale_factor * 0.9)
            elif processing_time < self.frame_time * 0.5:
                # Fast enough - increase size for better accuracy
                self.scale_factor = min(1.0, self.scale_factor * 1.1)
        
        # Apply resize
        height, width = frame.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        
        return cv2.resize(frame, (new_width, new_height))
    
    def adaptive_frame_skip(self, processing_time):
        """
        Dynamically adjust frame skipping based on processing time
        
        Args:
            processing_time: Time taken to process last frame
            
        Returns:
            int: Number of frames to skip
        """
        if processing_time > self.frame_time * 2:
            # Very slow - skip more frames
            self.skip_frames = min(5, self.skip_frames + 1)
        elif processing_time < self.frame_time * 0.8:
            # Fast enough - reduce skipping
            self.skip_frames = max(1, self.skip_frames - 1)
        
        return self.skip_frames
    
    def assess_frame_quality(self, frame):
        """
        Assess the quality of a frame for face recognition
        
        Args:
            frame: Input frame
            
        Returns:
            float: Quality score (0-1)
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)
        
        # Calculate brightness
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 127) / 127.0
        
        # Calculate contrast
        contrast = np.std(gray)
        contrast_score = min(1.0, contrast / 70.0)
        
        # Combine scores
        quality_score = (sharpness_score * 0.5 + 
                        brightness_score * 0.25 + 
                        contrast_score * 0.25)
        
        self.quality_scores.append(quality_score)
        
        return float(quality_score)
    
    def get_best_frame(self, frames):
        """
        Select the best quality frame from a list
        
        Args:
            frames: List of frames
            
        Returns:
            numpy.ndarray: Best quality frame
        """
        if not frames:
            return None
        
        best_frame = None
        best_score = -1
        
        for frame in frames:
            score = self.assess_frame_quality(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
        
        return best_frame
    
    def get_optimization_recommendations(self):
        """
        Get recommendations for system optimization
        
        Returns:
            dict: Optimization recommendations
        """
        recommendations = {}
        
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            
            if avg_time > self.frame_time * 1.5:
                recommendations['action'] = 'reduce_load'
                recommendations['suggestions'] = [
                    'Increase frame skipping',
                    'Reduce frame size',
                    'Use faster face detection model (hog instead of cnn)',
                    'Process every other frame'
                ]
            elif avg_time < self.frame_time * 0.5:
                recommendations['action'] = 'increase_quality'
                recommendations['suggestions'] = [
                    'Increase frame size for better accuracy',
                    'Reduce frame skipping',
                    'Use more accurate model'
                ]
            else:
                recommendations['action'] = 'optimal'
                recommendations['suggestions'] = [
                    'Current settings are optimal'
                ]
        
        if self.quality_scores:
            avg_quality = np.mean(self.quality_scores)
            if avg_quality < 0.5:
                recommendations['quality_issue'] = True
                recommendations['quality_suggestions'] = [
                    'Improve lighting conditions',
                    'Check camera focus',
                    'Reduce motion blur'
                ]
        
        return recommendations