# CMYK Registration & Tilt Analyzer

[![Build Windows](https://github.com/Diatomicc2/{repo}/workflows/Build%20Windows%20Executable/badge.svg)](https://github.com/Diatomicc2/{repo}/actions)
[![Build Status](https://github.com/Diatomicc2/{repo}/workflows/Build%20Executables/badge.svg)](https://github.com/Diatomicc2/{repo}/actions)

A professional GUI application for industrial printing quality management and calibration through CMYK color box alignment and tilt analysis.

## 🎯 Overview

This project provides an advanced solution for analyzing color registration accuracy and detecting tilt issues in industrial printing processes. It's designed to help printing professionals maintain high-quality output by automatically detecting and measuring misalignments in CMYK color registration marks.

## ✨ Features

### Core Analysis Capabilities
- **CMYK Color Detection**: Automatic detection of Cyan, Magenta, Yellow, and Special (K) color boxes
- **Registration Analysis**: Precise measurement of color box alignment and positioning
- **Tilt Detection**: Identification and measurement of angular misalignments
- **Perspective Correction**: Automatic image perspective transformation for accurate measurements

### User Interface
- **Modern GUI**: Built with PySide6 for a professional, cross-platform experience
- **Real-time Processing**: Live image analysis with immediate results
- **Batch Processing**: Support for multiple image analysis
- **Export Functionality**: Results export to CSV, Excel, and PDF formats
- **Debug Mode**: Comprehensive debugging tools with visual overlays

### Technical Features
- **Computer Vision**: Advanced OpenCV-based image processing
- **HSV Color Space**: Robust color detection using HSV color ranges
- **Contour Analysis**: Sophisticated shape detection and analysis
- **Coordinate Systems**: Precise pixel-to-coordinate conversion
- **Multi-threading**: Non-blocking UI with background processing

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

#### Option 1: Traditional Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd project3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run_gui.py
```

#### Option 2: Docker (Recommended for Easy Setup)
**Prerequisites:**
- Docker Desktop installed
- Docker Compose available

**Quick Start with Docker:**
```bash
# Clone the repository
git clone <repository-url>
cd project3

# Run with Docker (Linux/macOS)
./run_docker.sh

# Run with Docker (Windows)
run_docker.bat
```

**Manual Docker Commands:**
```bash
# Build and run
docker-compose up --build

# Or build first, then run
docker-compose build
docker-compose up
```

**Stop the application:**
```bash
docker-compose down
```

**Benefits of Docker:**
- ✅ No need to install Python or dependencies
- ✅ Works the same on all operating systems
- ✅ Easy to share and deploy
- ✅ Isolated environment
- ✅ No conflicts with system packages

#### Option 3: Standalone Executable
**Prerequisites:**
- Python 3.8+ installed (for building only)
- Windows, macOS, or Linux

**Quick Build (Auto-detect Platform):**
```bash
./build.sh
```

**Platform-Specific Builds:**
```bash
# macOS
./scripts/build.sh
python scripts/build_macos.py

# Windows (via Docker)
./scripts/build_windows_docker.sh
./scripts/build_windows_simple.sh

# Windows (native)
python scripts/build_windows.py

# PyInstaller direct
pyinstaller scripts/CMYK_Analyzer.spec
```

**Build Executable:**
```bash
# Windows
build_exe.bat

# Linux/macOS
./build_exe.sh

# Or manually with Python
python build_exe.py
```

**Use Executable:**
- After building, copy the `CMYK_Analyzer_Release/` folder to any computer
- Double-click `CMYK_Analyzer.exe` (Windows) or `CMYK_Analyzer` (Linux/macOS)
- No Python installation required on target computer

**Benefits of Executable:**
- ✅ No Python installation required on target computer
- ✅ Single file distribution
- ✅ Works offline
- ✅ Easy to share via USB or network
- ✅ Professional application feel

## 📋 Requirements

The following packages are required:

- `opencv-python>=4.8.0` - Computer vision and image processing
- `numpy>=1.24.0` - Numerical computing
- `PySide6>=6.5.0` - Modern Qt-based GUI framework
- `Pillow>=10.0.0` - Image processing
- `pandas>=2.0.0` - Data manipulation and analysis
- `openpyxl>=3.1.0` - Excel file handling
- `reportlab>=4.0.0` - PDF report generation

## 🎮 Usage

### Starting the Application
1. Run `python run_gui.py` from the project directory
2. The main GUI window will open with analysis options

### Basic Workflow
1. **Load Image**: Select an image containing CMYK registration marks
2. **Configure Analysis**: Set color detection parameters and analysis options
3. **Run Analysis**: Execute the registration and tilt analysis
4. **Review Results**: Examine detected color boxes and measurements
5. **Export Data**: Save results in your preferred format

### Advanced Features
- **Custom Color Ranges**: Adjust HSV values for specific color detection
- **Batch Processing**: Analyze multiple images simultaneously
- **Debug Mode**: Enable detailed processing information and visual overlays
- **Measurement Calibration**: Fine-tune measurement accuracy

## 🏗️ Architecture

### Core Modules

#### `cmyk_analyzer_gui.py`
Main GUI application with PySide6 interface, providing:
- Image loading and display
- Analysis parameter configuration
- Real-time processing controls
- Results visualization
- Export functionality

#### `color_registration_analysis.py`
Core analysis engine containing:
- `extract_marker()`: Image perspective correction
- `detect_bottom_left()`: Color box corner detection
- `detect_square_corners()`: Complete box corner analysis
- `pixel_to_bottom_left_coord()`: Coordinate conversion utilities

#### `run_gui.py`
Application launcher with:
- Dependency checking
- Environment validation
- Error handling
- Application startup

### Data Flow
1. **Image Input** → Raw image loading
2. **Preprocessing** → Perspective correction and enhancement
3. **Color Detection** → HSV-based color segmentation
4. **Shape Analysis** → Contour detection and validation
5. **Measurement** → Distance and angle calculations
6. **Results** → Data export and visualization

## 🔧 Configuration

### Color Detection Parameters
The system uses predefined HSV ranges for CMYK colors:
- **Cyan (C)**: Specific HSV range for cyan detection
- **Magenta (M)**: Specific HSV range for magenta detection
- **Yellow (Y)**: Specific HSV range for yellow detection
- **Special (K)**: Automatic detection based on box size analysis

### Analysis Settings
- **Minimum Area Ratio**: Filter for valid color boxes
- **Tolerance Levels**: Acceptable deviation thresholds
- **Debug Mode**: Enable detailed processing information

## 📊 Output Formats

### Data Export
- **CSV**: Tabular data for spreadsheet analysis
- **Excel**: Formatted reports with multiple sheets
- **PDF**: Professional reports with visual elements

### Analysis Results
- Color box coordinates and dimensions
- Registration accuracy measurements
- Tilt angle calculations
- Quality assessment scores
- Processing metadata

## 🐛 Troubleshooting

### Common Issues
1. **Missing Dependencies**: Ensure all requirements are installed
2. **Image Quality**: Use high-resolution images with clear color separation
3. **Color Detection**: Adjust HSV ranges for specific lighting conditions
4. **Performance**: Large images may require longer processing times

### Debug Mode
Enable debug mode to:
- View intermediate processing steps
- Identify detection failures
- Optimize parameters
- Validate analysis accuracy

## 🤝 Contributing

We welcome contributions to improve the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
---

**Note**: This application is designed for professional printing environments and requires proper calibration for optimal results.
