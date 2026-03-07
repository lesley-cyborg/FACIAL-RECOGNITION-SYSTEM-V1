#!/usr/bin/env python3
"""
Main entry point for the Facial Recognition System
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'recognize', 'realtime', 'api', 'batch'],
                       default='realtime', help='Operation mode')
    
    parser.add_argument('--system', type=str, 
                       choices=['basic', 'optimized', 'advanced', 'database', 'secure'],
                       default='basic', help='System type to use')
    
    parser.add_argument('--input', type=str, help='Input image path for recognition mode')
    
    parser.add_argument('--output', type=str, help='Output path for results')
    
    parser.add_argument('--name', type=str, help='Person name for registration')
    
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def create_system(system_type, config):
    """
    Create facial recognition system based on type
    
    Args:
        system_type: Type of system to create
        config: Configuration dictionary
        
    Returns:
        Facial recognition system instance
    """
    systems = {
        'basic': FacialRecognitionSystem,
        'optimized': OptimizedFacialRecognitionSystem,
        'advanced': AdvancedRecognitionSystem,
        'database': DatabaseFacialRecognition,
        'secure': SecureFacialRecognition
    }
    
    system_class = systems.get(system_type, FacialRecognitionSystem)
    
    # Extract paths from config dictionary
    storage_config = config.get('storage', {})
    encodings_path = storage_config.get('encodings_path', 'data/encodings/face_encodings.pkl')
    
    print(f"Using encodings path: {encodings_path}")
    
    # Pass appropriate parameters based on system type
    if system_type in ['database', 'secure']:
        db_path = storage_config.get('database_path', 'data/database/face_recognition.db')
        print(f"Using database path: {db_path}")
        return system_class(
            db_path=db_path,
            encodings_path=encodings_path
        )
    else:
        return system_class(
            encodings_path=encodings_path
        )


def train_mode(system, args, logger):
    """
    Handle training mode with both image and camera registration
    
    Args:
        system: Facial recognition system
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("Starting training mode")
    
    if args.name:
        if args.input:
            # Register from image file
            if os.path.exists(args.input):
                logger.info(f"Registering {args.name} from image: {args.input}")
                system.register_face_from_image(args.input, args.name)
            else:
                logger.error(f"Image file not found: {args.input}")
        else:
            # Register from camera
            logger.info(f"Registering {args.name} from camera (device {args.camera})")
            print("\n" + "="*50)
            print("CAMERA REGISTRATION MODE")
            print("="*50)
            print("Instructions:")
            print("1. Look at the camera")
            print("2. Press SPACE to capture your face")
            print("3. Press ESC to cancel")
            print("4. Try different angles for better recognition")
            print("="*50 + "\n")
            system.register_face_from_camera(args.name)
    else:
        logger.error("Training mode requires --name parameter")
        print("\nUsage: python main.py --mode train --name \"Your Name\" [--input image.jpg]")
        print("  - With --input: Register from image file")
        print("  - Without --input: Register from camera\n")


def recognize_mode(system, args, logger):
    """
    Handle recognition mode for images
    
    Args:
        system: Facial recognition system
        args: Command line arguments
        logger: Logger instance
    """
    logger.info(f"Starting recognition mode on {args.input}")
    
    if not args.input:
        logger.error("Recognition mode requires --input parameter")
        print("\nUsage: python main.py --mode recognize --input image.jpg\n")
        return
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Get tolerance from config if available
    tolerance = None
    if hasattr(system, 'tolerance'):
        tolerance = system.tolerance
    
    system.recognize_faces_from_image(args.input, tolerance)


def realtime_mode(system, args, logger):
    """
    Handle real-time recognition mode
    
    Args:
        system: Facial recognition system
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("Starting real-time recognition mode")
    
    if len(system.known_face_names) == 0:
        logger.warning("No known faces registered. Please register faces first.")
        print("\n" + "!"*50)
        print("WARNING: No faces registered!")
        print("Please register at least one face using:")
        print("  python main.py --mode train --name \"Your Name\"")
        print("!"*50 + "\n")
        
        # Ask user if they want to register now
        response = input("Would you like to register a face now? (y/n): ").strip().lower()
        if response == 'y':
            name = input("Enter your name: ").strip()
            if name:
                train_mode(system, argparse.Namespace(
                    name=name, 
                    input=None, 
                    camera=args.camera,
                    debug=args.debug
                ), logger)
            else:
                print("Registration cancelled.")
    
    # Start real-time recognition
    print("\n" + "="*50)
    print("REAL-TIME RECOGNITION MODE")
    print("="*50)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'd' to toggle debug info")
    print("="*50 + "\n")
    
    # Get tolerance from config if available
    tolerance = None
    if hasattr(system, 'tolerance'):
        tolerance = system.tolerance
    
    # Use appropriate real-time method based on system type
    if hasattr(system, 'secure_real_time_recognition'):
        system.secure_real_time_recognition(args.camera)
    elif hasattr(system, 'real_time_recognition'):
        system.real_time_recognition(tolerance, args.camera)
    else:
        logger.error(f"System {type(system).__name__} doesn't support real-time recognition")


def api_mode(system, args, logger, config):
    """
    Handle API server mode
    
    Args:
        system: Facial recognition system
        args: Command line arguments
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Starting API server mode")
    
    # Get API configuration
    api_config = config.get('api', {})
    host = api_config.get('host', '127.0.0.1')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False) or args.debug
    
    print("\n" + "="*50)
    print("API SERVER MODE")
    print("="*50)
    print(f"Server will start on: http://{host}:{port}")
    print("\nAvailable endpoints:")
    print("  - GET  /          - Web interface")
    print("  - GET  /health    - Health check")
    print("  - POST /recognize - Recognize faces")
    print("  - POST /register  - Register faces")
    print("  - GET  /faces     - List registered faces")
    print("  - GET  /stats     - System statistics")
    print("  - GET  /history   - Recognition history")
    print("="*50 + "\n")
    print("Press Ctrl+C to stop the server\n")
    
    # Create and run API
    api = FacialRecognitionAPI(system)
    api.run(host=host, port=port, debug=debug)


def batch_mode(system, args, logger):
    """
    Handle batch processing mode
    
    Args:
        system: Facial recognition system
        args: Command line arguments
        logger: Logger instance
    """
    logger.info("Starting batch processing mode")
    
    if not args.input:
        logger.error("Batch mode requires --input folder path")
        print("\nUsage: python main.py --mode batch --input /path/to/images/folder\n")
        return
    
    if not os.path.exists(args.input):
        logger.error(f"Input folder not found: {args.input}")
        return
    
    batch_processor = BatchFacialRecognition(system)
    
    print("\n" + "="*50)
    print("BATCH PROCESSING MODE")
    print("="*50)
    print(f"Processing folder: {args.input}")
    print("="*50 + "\n")
    
    # Ask what operation to perform
    print("Select operation:")
    print("1. Register faces from folder")
    print("2. Recognize faces in images")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        # Batch registration
        mapping_file = input("Enter path to name mapping CSV (optional, press Enter to skip): ").strip()
        if mapping_file and not os.path.exists(mapping_file):
            logger.warning(f"Mapping file not found: {mapping_file}")
            mapping_file = None
        
        registered, failed = batch_processor.batch_register_from_folder(
            args.input,
            mapping_file if mapping_file else None,
            recursive=True
        )
        
        print(f"\nBatch registration complete!")
        print(f"Successfully registered: {len(registered)}")
        print(f"Failed: {len(failed)}")
        
    elif choice == '2':
        # Batch recognition
        output_path = args.output or f"reports/batch_results_{utils.get_timestamp()}.json"
        results = batch_processor.batch_recognize_from_folder(args.input, output_path)
        
        print(f"\nBatch recognition complete!")
        print(f"Processed {len(results)} images")
        print(f"Results saved to: {output_path}")
    else:
        logger.error("Invalid choice")


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    print_banner()
    
    # Setup directories
    setup_directories()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Debug: print config info if debug mode
    if args.debug:
        print(f"Config type: {type(config)}")
        if isinstance(config, dict):
            print(f"Config keys: {list(config.keys())}")
    
    # Setup logging
    logger = setup_logging(config)
    
    try:
        # Create system
        logger.info(f"Creating {args.system} facial recognition system")
        system = create_system(args.system, config)
        
        # Execute based on mode
        if args.mode == 'train':
            train_mode(system, args, logger)
            
        elif args.mode == 'recognize':
            recognize_mode(system, args, logger)
            
        elif args.mode == 'realtime':
            realtime_mode(system, args, logger)
            
        elif args.mode == 'api':
            api_mode(system, args, logger, config)
            
        elif args.mode == 'batch':
            batch_mode(system, args, logger)
        
        # Print summary if in train mode
        if args.mode == 'train' and args.name:
            print(f"\n✓ Registration complete!")
            print(f"Total registered faces: {system.get_face_count()}")
            if system.get_face_count() > 0:
                print("\nRegistered people:")
                for i, name in enumerate(system.known_face_names, 1):
                    print(f"  {i}. {name}")
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        print("\n\nShutting down...")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())