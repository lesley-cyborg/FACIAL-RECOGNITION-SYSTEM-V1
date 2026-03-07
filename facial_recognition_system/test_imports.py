# test_imports.py
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    from src.base_system import FacialRecognitionSystem
    print("✓ base_system")
except ImportError as e:
    print(f"✗ base_system: {e}")

try:
    from src.optimized_system import OptimizedFacialRecognitionSystem
    print("✓ optimized_system")
except ImportError as e:
    print(f"✗ optimized_system: {e}")

try:
    from src.advanced_recognition import AdvancedRecognitionSystem
    print("✓ advanced_recognition")
except ImportError as e:
    print(f"✗ advanced_recognition: {e}")

try:
    from src.database_system import DatabaseFacialRecognition
    print("✓ database_system")
except ImportError as e:
    print(f"✗ database_system: {e}")

try:
    from src.secure_system import SecureFacialRecognition
    print("✓ secure_system")
except ImportError as e:
    print(f"✗ secure_system: {e}")

print("\nAll tests complete!")