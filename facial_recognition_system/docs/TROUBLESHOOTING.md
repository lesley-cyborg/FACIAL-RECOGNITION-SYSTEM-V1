
### `docs/TROUBLESHOOTING.md`
```markdown
# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. dlib Installation Fails

**Error:**

Failed building wheel for dlib
error: command 'x86_64-linux-gnu-gcc' failed


**Solutions:**

**Ubuntu/Debian:**
```bash
# Install build dependencies
sudo apt update
sudo apt install -y build-essential cmake
sudo apt install -y libx11-dev libgtk-3-dev
sudo apt install -y libboost-python-dev libboost-thread-dev
sudo apt install -y libopenblas-dev liblapack-dev

# Try installing dlib separately
pip install dlib --no-cache-dir

#**Windows:**

# Install Visual Studio Build Tools
#Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# Select "C++ build tools" during installation

# Then install dlib
pip install dlib

# Navigate to your project directory first
cd C:\Users\kagis\OneDrive\Desktop\facial_recognition_system

# Install all dependencies from requirements.txt
pip install -r requirements.txt

## macOS:

bash
# Install cmake
brew install cmake
brew install boost

# Install dlib
pip install dlib