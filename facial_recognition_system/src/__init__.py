# src/__init__.py
"""
Facial Recognition System Package
"""

__version__ = "2.0.0"

# Import all the main classes for easy access
from src.base_system import FacialRecognitionSystem
from src.optimized_system import OptimizedFacialRecognitionSystem
from src.advanced_recognition import AdvancedRecognitionSystem
from src.database_system import DatabaseFacialRecognition
from src.secure_system import SecureFacialRecognition
from src.api_server import FacialRecognitionAPI
from src.realtime_optimizer import RealTimeOptimizer
from src.batch_processor import BatchFacialRecognition
from src.config_manager import ConfigManager
from src.utils import setup_directories, setup_logging, print_banner

__all__ = [
    'FacialRecognitionSystem',
    'OptimizedFacialRecognitionSystem',
    'AdvancedRecognitionSystem',
    'DatabaseFacialRecognition',
    'SecureFacialRecognition',
    'FacialRecognitionAPI',
    'RealTimeOptimizer',
    'BatchFacialRecognition',
    'ConfigManager',
    'setup_directories',
    'setup_logging',
    'print_banner'
]