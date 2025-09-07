# Banana Vision 🛰️

A comprehensive toolkit for converting NetCDF satellite data (Landsat/MODIS) and AI embeddings into organized image datasets, with advanced interpolation capabilities for climate and agricultural monitoring.

## Overview

This project transforms complex satellite reflectance data and AI-generated embeddings into visual formats suitable for analysis, machine learning, and presentation. Originally designed for grassland monitoring and agricultural applications, it supports both standard satellite bands and AlphaEarth embeddings.

> **Why "Banana Vision"?** 🍌 The project was originally developed to analyze agricultural land use patterns, where the characteristic yellow-to-green color transitions in satellite imagery reminded us of ripening bananas - a perfect metaphor for tracking vegetation changes over time!

## Features

- **Automatic band detection**: Creates RGB and false-color composites for reflectance data
- **Date-based filenames**: Uses actual acquisition dates instead of indices
- **Organized output**: Images sorted into subdirectories by band/channel type
- **Multiple formats**: PNG, JPEG, TIFF support
- **Flexible time selection**: Process specific time steps or entire time series

## Prerequisites

- Python 3.7+
- NetCDF4 library support
- Optional: Gemini API key for AI-powered interpolation

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/banana-vision.git
   cd banana-vision
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key (optional, for interpolation)**
   ```bash
   echo "GEMINI_API_KEY=your_key_here" > .env
   ```
   Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Convert single time step (default: first)
python nc_to_image.py input/landsat_landsat_US-Cop.nc

# Convert ALL time steps (WARNING: creates thousands of images!)
python nc_to_image.py input/landsat_landsat_US-Cop.nc --all-times

# Convert specific time steps with custom output
python nc_to_image.py input/landsat_landsat_US-Cop.nc -t 0 10 20 30 -o my_images

# Convert to JPEG format
python nc_to_image.py input/landsat_landsat_US-Cop.nc -f jpeg --all-times
```

## Output Structure

```
output/
├── rgb/                    # RGB composite images (natural color)
├── false_color/           # False color composites (vegetation analysis)
├── red/                   # Red band (630-680nm)
├── green/                 # Green band (525-600nm)
├── blue/                  # Blue band (450-515nm)
├── nir08/                 # Near-infrared (845-885nm) - KEY for vegetation
├── swir16/                # Short-wave infrared band 1
├── swir22/                # Short-wave infrared band 2
├── qa_pixel/              # Quality assessment
└── embeddings/            # AlphaEarth embedding bands (if present)
```

## Use Cases

### 🌱 Agricultural & Grassland Monitoring
Track vegetation health, crop growth patterns, and land use changes over time.
```bash
# Extract key bands for vegetation analysis
python nc_to_image.py input/landsat_landsat_US-Cop.nc --all-times -o grassland_analysis

# Focus on growing seasons (adjust indices for your region)
python nc_to_image.py input/landsat_landsat_US-Cop.nc -t 100 200 300 400 500
```

### 🔍 Quick Visual Assessment
Generate representative samples for presentations or initial analysis.
```bash
# RGB and false-color composites only
python nc_to_image.py input/landsat_landsat_US-Cop.nc -t 0 50 100 150 200
```

### 🤖 Machine Learning Datasets
Create training data for computer vision models in remote sensing.
```bash
# Generate comprehensive dataset with all bands
python nc_to_image.py input/landsat_landsat_US-Cop.nc --all-times -o ml_dataset

# Convert AlphaEarth AI embeddings for analysis
python nc_to_image.py input/embeddings_alpha_earth_US-Cop.nc --all-times -o ai_embeddings
```

### 📊 Time Series Analysis
Build smooth temporal progressions for change detection studies.
```bash
# 1. Extract keyframe images
python nc_to_image.py input/landsat_landsat_US-Cop.nc -t 0 100 200

# 2. Generate interpolated frames between keyframes
python interpolate_images.py output/rgb/image_2002-01-12.png output/rgb/image_2004-01-15.png
```

### 🌍 Climate Change Research
Monitor environmental changes and create compelling visualizations.
```bash
# Extract specific bands critical for climate analysis
python nc_to_image.py input/landsat_landsat_US-Cop.nc --all-times -o climate_study
```

## File Naming Convention

Images are named with coordinates and acquisition dates:
- `landsat_landsat_US-Cop_x641190_y4217025_rgb_2002-01-12.png`
- `landsat_landsat_US-Cop_x641190_y4217025_nir08_2024-09-29.png`
- `embeddings_alpha_earth_US-Cop_x641190_y4217025_embedding_b00_2017-01-01.png`

**Note**: All files share the same coordinates `x641190_y4217025`, indicating this dataset tracks a **single geographic location** over time (time series analysis) rather than multiple spatial locations.

## Key Bands for Analysis

| Band | Best For |
|------|----------|
| **false_color/** | Vegetation health assessment (healthy = bright red) |
| **nir08/** | Biomass detection, NDVI calculation |
| **red/** | NDVI calculation, plant stress |
| **rgb/** | Visual interpretation, presentations |

## Command Options

```
usage: nc_to_image.py [-h] [-o OUTPUT] [-f {png,jpg,jpeg,tiff}]
                      [-t TIME_INDICES [TIME_INDICES ...]] [--all-times]
                      [--no-composites]
                      input

positional arguments:
  input                 Input NetCDF file or directory

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output directory (default: ./output)
  -f {png,jpg,jpeg,tiff}, --format {png,jpg,jpeg,tiff}
                        Output image format (default: png)
  -t TIME_INDICES [TIME_INDICES ...]
                        Time indices to process (default: first time step only)
  --all-times           Process all time steps (WARNING: can create thousands of images!)
  --no-composites       Skip creating RGB/false-color composites
```

## Dataset Information

| Dataset Type | Time Steps | Bands | Total Images | Time Range | Location |
|--------------|------------|-------|--------------|------------|----------|
| **Landsat** | 778 | 14 | ~10,892 | 2002-2024 | Single pixel (x641190_y4217025) |
| **AlphaEarth** | 8 | 5 | ~40 | 2017-2024 | Single pixel (x641190_y4217025) |

**Important**: This dataset represents **time series data for one geographic location**, not spatial coverage. Perfect for:
- Monitoring vegetation changes at a specific site over 22+ years
- Analyzing seasonal patterns and long-term trends
- Creating time-lapse visualizations of land use changes

### Performance Notes
- **Full extraction** can take 30+ minutes for large datasets
- **Memory usage** scales with image resolution and band count  
- **Disk space**: ~2-5GB for complete Landsat extraction
- **Recommended**: Start with specific time ranges (`-t`) before using `--all-times`

## Image Interpolation

Use `interpolate_images.py` to generate missing images between two dates using AI:

```bash
# Set up your Gemini API key in .env file first
# Get free key from: https://aistudio.google.com/app/apikey

# Binary recursive interpolation
python interpolate_images.py image_2002-01-12.png image_2002-03-01.png

# Binary interpolation generates middle first, then 1/4, 3/4, etc.
# Order: middle → quarters → eighths → sixteenths...
# More balanced and intelligent progression

# Simple interpolation without AI (fastest)
python interpolate_images.py start.png end.png --simple-only
```

### AI-Enhanced Cloud Handling

The interpolation AI has been specifically trained to understand satellite imagery characteristics:

- **Cloud Recognition**: Automatically identifies black pixels as clouds or data gaps
- **Clear Ground Priority**: Favors showing actual ground features over cloud patterns
- **Smart Interpolation**: When one image has clouds and another is clear, it prioritizes the clear ground data
- **Realistic Results**: Generates plausible ground cover in previously clouded areas

### Interpolation Method

**Binary Recursive Interpolation**: Generates the middle frame first, then recursively fills in quarter points, eighth points, etc. This creates a more balanced and intelligent progression compared to sequential day-by-day generation.

**Example**: Between Jan 12 and Mar 1 (48 days apart), it generates 47 intermediate images for every missing day!

**Note**: The interpolation script automatically copies the original start and end images to the output folder, so you'll have a complete sequence including the originals.

## Dependencies

All required packages are listed in `requirements.txt`:
- **Core data processing**: `xarray`, `netcdf4`, `numpy`, `pandas`
- **Image processing**: `matplotlib`, `pillow`
- **AI interpolation**: `google-generativeai`, `python-dotenv`
- **Web requests**: `requests`

## Troubleshooting

### Common Issues

**NetCDF file not found**
```bash
# Ensure your NetCDF files are in the input/ directory
ls input/*.nc
```

**API key issues for interpolation**
```bash
# Check if .env file exists and contains your key
cat .env
# Should show: GEMINI_API_KEY=your_actual_key_here
```

**Memory issues with large datasets**
```bash
# Process smaller time ranges instead of --all-times
python nc_to_image.py input/file.nc -t 0 50 100  # Process 3 time steps
```

**Python environment issues**
```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python
pip list | grep xarray  # Verify packages are installed
```

## Project Structure

```
banana-vision/
├── input/                     # NetCDF data files
├── output/                    # Generated images (default)
│   ├── rgb/                   # RGB composite images
│   ├── false_color/           # False color composites
│   ├── [band_folders]/        # Individual band outputs
│   └── full_dataset_extract/  # Complete dataset extractions
├── nc_to_image.py            # Main conversion script
├── interpolate_images.py     # AI-powered interpolation
├── requirements.txt          # Python dependencies
├── venv/                     # Virtual environment
└── README.md                 # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for climate and agricultural monitoring applications
- Supports Landsat and MODIS satellite data
- Integration with AlphaEarth AI embeddings
- Powered by Google's Gemini AI for interpolation