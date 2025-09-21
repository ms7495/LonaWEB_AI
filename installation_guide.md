# LonaWEB AI - Installation Guide

## Quick Start Installation

### 1. Basic Installation
```bash
# Install core requirements
pip install -r requirements.txt
```

### 2. GPU-Accelerated Installation (Recommended for NVIDIA GPUs)
```bash
# Install core requirements first
pip install -r requirements.txt

# Uninstall CPU-only llama-cpp-python
pip uninstall llama-cpp-python -y

# Install GPU-accelerated version
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### 3. Alternative GPU Installation (CUDA 11.8)
```bash
# For older CUDA versions
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

## Minimal Installation (Core Only)

If you want to install only the essential dependencies:

```bash
# Create minimal requirements
pip install streamlit fastapi uvicorn sentence-transformers qdrant-client pdfplumber python-docx pandas numpy
```

## Development Installation

For development and testing:

```bash
# Install all requirements including dev tools
pip install -r requirements.txt

# Or install with development extras
pip install -r requirements-dev.txt
```

## Docker Installation (Alternative)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# For GPU support in Docker
# FROM nvidia/cuda:11.8-devel-ubuntu20.04
```

## Platform-Specific Notes

### Windows
```bash
# May need Visual Studio Build Tools for some packages
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install requirements
pip install -r requirements.txt
```

### macOS
```bash
# Install Xcode Command Line Tools first
xcode-select --install

# Install requirements
pip install -r requirements.txt

# For M1/M2 Macs, use Metal acceleration
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

### Linux
```bash
# Install system dependencies first
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Install requirements
pip install -r requirements.txt
```

## Verification

After installation, verify everything works:

```python
# Test script
import streamlit
import sentence_transformers
import qdrant_client
import pdfplumber
import pandas
print("✅ All core dependencies installed successfully!")
```

## Troubleshooting

### Common Issues

1. **llama-cpp-python compilation errors**:
   ```bash
   pip install --upgrade pip wheel setuptools
   pip install llama-cpp-python --no-cache-dir
   ```

2. **CUDA not detected**:
   ```bash
   # Check CUDA installation
   nvidia-smi
   nvcc --version
   ```

3. **Memory issues**:
   ```bash
   # Install with limited memory usage
   pip install --no-cache-dir -r requirements.txt
   ```

4. **Permission errors**:
   ```bash
   # Use user installation
   pip install --user -r requirements.txt
   ```

## Environment Setup

Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv lonaweb_env

# Activate (Windows)
lonaweb_env\Scripts\activate

# Activate (macOS/Linux)
source lonaweb_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Optional Enhancements

### Spacy Language Models
```bash
# Download English language model
python -m spacy download en_core_web_sm
```

### NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Additional Models
```bash
# Download specific embedding models
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```