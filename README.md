# FACIAL-RECOGNITION-SYSTEM-V1
First attempt to making a real-time face recognition system like in the movies.....

## 📸 Overview

A comprehensive, production-ready facial recognition system built in Python. This system provides real-time face detection, recognition, and management capabilities with support for multiple system configurations, database integration, security features, and a RESTful API.

### 🎯 Key Features

- **Real-time Face Recognition** - Live camera feed recognition with optimized performance
- **Multiple Recognition Modes** - Image, video, and real-time processing
- **Advanced Security** - Password authentication, JWT tokens, anti-spoofing measures
- **Database Integration** - SQLite support with recognition logging and access control
- **RESTful API** - Full API for remote access and integration
- **Batch Processing** - Process multiple images simultaneously
- **Multi-Face Registration** - Register faces from images, camera, or video files
- **Performance Optimized** - Async processing, frame skipping, adaptive resizing
- **Comprehensive Logging** - Detailed logs for debugging and monitoring
- **Web Interface** - User-friendly dashboard for all operations

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Security Features](#security-features)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- 4GB RAM minimum (8GB recommended)
- Windows 10/11, Ubuntu 20.04+, or macOS 10.15+

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/facial_recognition_system.git
cd facial_recognition_system
2. Set Up Virtual Environment
Windows (PowerShell):

powershell
python -m venv venv
.\venv\Scripts\activate
Windows (Command Prompt):

cmd
python -m venv venv
venv\Scripts\activate.bat
Linux/Mac:

bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
4. Install Visual C++ Build Tools (Windows Only)
If you're on Windows, you need Visual C++ build tools for dlib:

Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

Run installer and select "Desktop development with C++"

Complete installation and restart your computer

5. Download Facial Landmark Model
bash
# Create models directory
mkdir models

# Download the model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
Windows PowerShell alternative:

powershell
$url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
$output = "models\shape_predictor_68_face_landmarks.dat.bz2"
Invoke-WebRequest -Uri $url -OutFile $output
# Extract using 7-Zip or similar
6. Verify Installation
bash
python -c "import face_recognition; print('✓ Installation successful')"
🏃 Quick Start
Register a Face
From Camera:

bash
python main.py --mode train --name "Your Name"
From Image:

bash
python main.py --mode train --name "Your Name" --input "path/to/your/photo.jpg"
Start Real-Time Recognition
bash
python main.py --mode realtime --system basic
Launch Web Interface
bash
python main.py --mode api --system basic
Then open http://localhost:5000 in your browser

🏗 System Architecture
text
facial_recognition_system/
├── main.py                 # Main entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── src/                    # Source code
│   ├── base_system.py      # Core recognition functionality
│   ├── optimized_system.py # Performance-optimized version
│   ├── advanced_recognition.py # Advanced features with landmarks
│   ├── database_system.py  # Database integration
│   ├── secure_system.py    # Security features
│   ├── api_server.py       # REST API
│   ├── realtime_optimizer.py # Performance optimization
│   ├── batch_processor.py  # Batch processing
│   ├── config_manager.py   # Configuration management
│   └── utils.py            # Utility functions
├── tests/                  # Unit tests
├── data/                   # Data storage
│   ├── encodings/          # Face encodings
│   ├── database/           # SQLite database
│   └── logs/               # Log files
├── models/                 # Pre-trained models
├── templates/              # Web interface templates
└── scripts/                # Utility scripts
📖 Usage Guide
Command Line Interface
bash
python main.py [--mode MODE] [--system SYSTEM] [--options]
Modes
Mode	Description	Example
train	Register new faces	--mode train --name "John"
recognize	Recognize faces in image	--mode recognize --input image.jpg
realtime	Real-time camera recognition	--mode realtime
api	Start API server	--mode api
batch	Batch process images	--mode batch --input ./photos
System Types
System	Description	Use Case
basic	Core functionality	Quick testing, minimal setup
optimized	Performance optimizations	Real-time processing, high FPS
advanced	Landmark detection	Detailed face analysis
database	Database integration	Production systems with logging
secure	Security features	Authentication required
Common Options
--config FILE - Custom config file (default: config.yaml)

--name NAME - Person name for registration

--input PATH - Input image/video/folder path

--output PATH - Output file path

--camera ID - Camera device ID (default: 0)

--debug - Enable debug mode

Examples
bash
# Register from camera
python main.py --mode train --name "Alice"

# Register from image
python main.py --mode train --name "Bob" --input "bob_photo.jpg"

# Real-time with optimized system
python main.py --mode realtime --system optimized

# Secure real-time with authentication
python main.py --mode realtime --system secure

# Recognize faces in image
python main.py --mode recognize --input "group_photo.jpg"

# Start API server
python main.py --mode api --system database

# Batch register from folder
python main.py --mode batch --input "training_faces/"
Web Interface
After starting the API server, access the web interface at http://localhost:5000:

Recognition Tab - Upload images or use camera for real-time recognition

Registration Tab - Register new faces with optional details

Management Tab - View and manage registered faces

Statistics Tab - View system statistics and recognition history

🌐 API Documentation
Base URL
text
http://localhost:5000/api
Endpoints
Health Check
bash
GET /api/health
Response:

json
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00",
    "version": "2.0.0"
}
Recognize Faces
bash
POST /api/recognize
Content-Type: application/json

{
    "image": "base64_encoded_image_data",
    "tolerance": 0.6
}
Response:

json
{
    "success": true,
    "faces_detected": 2,
    "results": [
        {
            "location": {
                "top": 100,
                "right": 200,
                "bottom": 300,
                "left": 400
            },
            "matches": [
                {
                    "name": "John Doe",
                    "confidence": 0.95
                }
            ]
        }
    ]
}
Register Face
bash
POST /api/register
Content-Type: application/json

{
    "image": "base64_encoded_image_data",
    "name": "John Doe",
    "email": "john@example.com",
    "department": "Engineering"
}
Response:

json
{
    "success": true,
    "message": "Registered John Doe successfully"
}
List Registered Faces
bash
GET /api/faces
Response:

json
{
    "success": true,
    "total_faces": 2,
    "faces": [
        {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
            "department": "Engineering"
        }
    ]
}
Delete Face
bash
DELETE /api/faces/{name}
Response:

json
{
    "success": true,
    "message": "Deleted John Doe"
}
Get Statistics
bash
GET /api/stats
Response:

json
{
    "success": true,
    "stats": {
        "total_faces": 10,
        "total_recognitions": 150,
        "recognitions_today": 25
    }
}
Get Recognition History
bash
GET /api/history?person=John&limit=100
Response:

json
{
    "success": true,
    "total": 50,
    "history": [
        {
            "name": "John Doe",
            "timestamp": "2024-01-01 12:00:00",
            "confidence": 0.95
        }
    ]
}
Export Data
bash
GET /api/export?format=json
Response: JSON or CSV file download

⚙️ Configuration
config.yaml
yaml
system:
  name: "Facial Recognition System"
  environment: "development"  # development, testing, production

recognition:
  tolerance: 0.6  # Recognition threshold (0-1)
  model: "hog"    # hog (faster) or cnn (more accurate)
  upsample_times: 1
  jitter: 1
  min_face_size: [80, 80]
  max_face_size: [300, 300]

performance:
  target_fps: 30
  frame_scale: 0.5
  use_gpu: false
  batch_size: 32
  num_workers: 4
  async_processing: true
  frame_skip: 2

storage:
  encodings_path: "data/encodings/face_encodings.pkl"
  database_path: "data/database/face_recognition.db"
  log_path: "data/logs/recognition.log"
  text_log_path: "data/logs/recognition_log.txt"
  exports_path: "data/exports/"

security:
  require_authentication: false
  anti_spoofing: false
  min_confidence: 0.4
  jwt_expiration_hours: 24
  max_login_attempts: 5
  lockout_minutes: 15

database:
  type: "sqlite"  # sqlite, mysql, postgresql
  pool_size: 10
  backup_interval_days: 7

camera:
  device_id: 0
  width: 640
  height: 480
  fps: 30
  brightness: 100
  contrast: 100

api:
  enabled: true
  host: "127.0.0.1"
  port: 5000
  debug: false
  cors_origins: ["http://localhost:3000"]
  rate_limit: "100/hour"

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "detailed"  # simple, detailed, json
  max_file_size_mb: 100
  backup_count: 5

monitoring:
  enabled: true
  metrics_port: 8000
  health_check_interval: 30

batch_processing:
  max_images_per_batch: 1000
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp"]
  output_format: "json"

training:
  validation_split: 0.2
  test_split: 0.1
  augment_data: true
  random_seed: 42
🔒 Security Features
Authentication System
Password hashing with bcrypt

JWT token-based authentication

Role-based access control (admin/manager/user/guest)

Login attempt tracking and lockout

Anti-Spoofing Measures
Blink detection for liveness

Texture analysis (detects printed photos)

Motion detection (detects static images)

Brightness and contrast analysis

Secure Configuration
Environment variables for sensitive data

Configurable rate limiting

CORS protection for API

SQL injection prevention

⚡ Performance Tuning
Optimization Tips
Adjust Frame Scaling

yaml
performance:
  frame_scale: 0.25  # Lower = faster but less accurate
Increase Frame Skipping

yaml
performance:
  frame_skip: 3  # Process every 3rd frame
Use HOG Model

yaml
recognition:
  model: "hog"  # Faster than CNN
Enable GPU Acceleration

yaml
performance:
  use_gpu: true
Batch Processing Settings

yaml
batch_processing:
  max_images_per_batch: 500  # Adjust based on memory
🔧 Troubleshooting
Common Issues and Solutions
dlib Installation Fails
Error: Failed building wheel for dlib
Solution: Install Visual C++ Build Tools (Windows) or build-essential (Linux)

Camera Not Detected
bash
# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
Import Errors
Error: ImportError: attempted relative import with no known parent package
Solution: Run from project root or set PYTHONPATH:

bash
export PYTHONPATH="${PYTHONPATH}:/path/to/facial_recognition_system"
Low Recognition Accuracy
Adjust tolerance in config (0.4-0.6)

Improve lighting conditions

Register multiple angles of the same person

Use higher quality images

High CPU Usage
Reduce target FPS

Increase frame skipping

Lower frame scale

Disable unused features

🧪 Testing
Run the test suite:

bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_base_system.py

# Run with coverage
python -m pytest --cov=src tests/
📦 Project Structure Details
Data Directories
data/encodings/ - Stored face encodings (pickle format)

data/database/ - SQLite database files

data/logs/ - Application logs

data/exports/ - Exported data

Output Directories
captured_frames/ - Saved frames from recognition

reports/ - Batch processing reports

Model Files
models/shape_predictor_68_face_landmarks.dat - Facial landmark detector

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

Development Guidelines
Follow PEP 8 style guide

Add docstrings for new functions

Include unit tests for new features

Update documentation as needed

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👏 Acknowledgments
dlib - C++ library for machine learning

face_recognition - Python face recognition library

OpenCV - Computer vision library

Flask - Web framework

📞 Support
For issues and questions:

Open an issue on GitHub

Check the Troubleshooting Guide

Review API Documentation

See Deployment Guide for production setup

Built with ❤️ by Kagiso Setilo aka Ninja
