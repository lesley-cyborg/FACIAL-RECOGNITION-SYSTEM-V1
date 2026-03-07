"""
Configuration management for facial recognition system
"""

import os
import yaml
import json
from pathlib import Path


class ConfigManager:
    """
    Manages system configuration
    """
    
    def __init__(self, config_path=None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or 'config.yaml'
        self.config = self.load_config()
        
    def load_config(self):
        """
        Load configuration from file
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            'system': {
                'name': 'Facial Recognition System',
                'version': '2.0.0',
                'environment': 'production'
            },
            'recognition': {
                'tolerance': 0.6,
                'model': 'hog',
                'upsample_times': 1,
                'jitter': 1,
                'min_face_size': [80, 80],
                'max_face_size': [300, 300]
            },
            'performance': {
                'target_fps': 30,
                'frame_scale': 0.5,
                'use_gpu': False,
                'batch_size': 32,
                'num_workers': 4,
                'async_processing': True,
                'frame_skip': 2
            },
            'storage': {
                'encodings_path': 'data/encodings/face_encodings.pkl',
                'database_path': 'data/database/face_recognition.db',
                'log_path': 'data/logs/recognition.log',
                'text_log_path': 'data/logs/recognition_log.txt',
                'exports_path': 'data/exports/'
            },
            'security': {
                'require_authentication': True,
                'anti_spoofing': True,
                'min_confidence': 0.4,
                'jwt_expiration_hours': 24,
                'max_login_attempts': 5,
                'lockout_minutes': 15
            },
            'database': {
                'type': 'sqlite',
                'pool_size': 10,
                'backup_interval_days': 7
            },
            'camera': {
                'device_id': 0,
                'width': 640,
                'height': 480,
                'fps': 30,
                'brightness': 100,
                'contrast': 100
            },
            'api': {
                'enabled': True,
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'cors_origins': ['http://localhost:3000'],
                'rate_limit': '100/hour'
            },
            'logging': {
                'level': 'INFO',
                'format': 'detailed',
                'max_file_size_mb': 100,
                'backup_count': 5
            },
            'monitoring': {
                'enabled': True,
                'metrics_port': 8000,
                'health_check_interval': 30
            },
            'batch_processing': {
                'max_images_per_batch': 1000,
                'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp'],
                'output_format': 'json'
            },
            'training': {
                'validation_split': 0.2,
                'test_split': 0.1,
                'augment_data': True,
                'random_seed': 42
            }
        }
        
        # Try to load user config
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                    
                # Deep merge configs
                config = self._deep_merge(default_config, user_config)
                print(f"Loaded configuration from {self.config_path}")
                return config
                
            except Exception as e:
                print(f"Error loading config: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _deep_merge(self, base, update):
        """
        Deep merge two dictionaries
        
        Args:
            base: Base dictionary
            update: Update dictionary
            
        Returns:
            dict: Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, *keys, default=None):
        """
        Get configuration value by keys
        
        Args:
            *keys: Key path
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, value, *keys):
        """
        Set configuration value
        
        Args:
            value: Value to set
            *keys: Key path
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, path=None):
        """
        Save configuration to file
        
        Args:
            path: Output path (defaults to config_path)
        """
        save_path = path or self.config_path
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                else:
                    json.dump(self.config, f, indent=2)
            
            print(f"Configuration saved to {save_path}")
            
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def validate(self):
        """
        Validate configuration
        
        Returns:
            tuple: (is_valid, errors)
        """
        errors = []
        
        # Check required paths
        required_dirs = [
            os.path.dirname(self.get('storage', 'encodings_path')),
            os.path.dirname(self.get('storage', 'database_path')),
            os.path.dirname(self.get('storage', 'log_path')),
            self.get('storage', 'exports_path')
        ]
        
        for dir_path in required_dirs:
            if dir_path and not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {dir_path}: {e}")
        
        # Validate numeric values
        if self.get('recognition', 'tolerance') < 0 or self.get('recognition', 'tolerance') > 1:
            errors.append("Tolerance must be between 0 and 1")
        
        if self.get('performance', 'target_fps') <= 0:
            errors.append("Target FPS must be positive")
        
        if self.get('camera', 'device_id') < 0:
            errors.append("Camera device ID must be non-negative")
        
        return len(errors) == 0, errors