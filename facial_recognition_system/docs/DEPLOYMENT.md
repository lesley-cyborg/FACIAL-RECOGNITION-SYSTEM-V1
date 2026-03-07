# Deployment Guide

## System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 10 GB free space
- **Camera**: USB webcam or built-in camera
- **OS**: Ubuntu 20.04+, Windows 10+, macOS 10.15+

### Recommended Requirements
- **CPU**: Quad-core 3.0 GHz or better
- **RAM**: 8 GB or more
- **GPU**: NVIDIA GPU with CUDA support (for faster processing)
- **Storage**: 20 GB SSD
- **Camera**: 1080p camera with good low-light performance

## Installation Options

### Option 1: Local Installation

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-dev build-essential
sudo apt install -y cmake libopencv-dev
sudo apt install -y libx11-dev libgtk-3-dev
sudo apt install -y libboost-all-dev

# Clone repository
git clone https://github.com/lesley-cyborg/facial_recognition_system.git
cd facial_recognition_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create models directory
mkdir models -Force

# Download the landmark predictor
# You can do this manually or use PowerShell:
$url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
$output = "models\shape_predictor_68_face_landmarks.dat.bz2"
Invoke-WebRequest -Uri $url -OutFile $output

# Extract the file (requires 7zip or similar, or do it manually)
# If you have 7zip installed:
& "C:\Program Files\7-Zip\7z.exe" x $output -omodels\

# Download facial landmark model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/

# Create directories
python -c "from src.utils import setup_directories; setup_directories()"

# Test installation
python main.py --mode realtime