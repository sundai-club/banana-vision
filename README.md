# NetCDF to Image Converter

Convert NetCDF files containing satellite reflectance data (Landsat/MODIS) or embeddings to organized image datasets.

## Features

- **Automatic band detection**: Creates RGB and false-color composites for reflectance data
- **Date-based filenames**: Uses actual acquisition dates instead of indices
- **Organized output**: Images sorted into subdirectories by band/channel type
- **Multiple formats**: PNG, JPEG, TIFF support
- **Flexible time selection**: Process specific time steps or entire time series

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Convert single time step (default: first)
python nc_to_image.py landsat_landsat_US-Cop.nc

# Convert ALL time steps (WARNING: creates thousands of images!)
python nc_to_image.py landsat_landsat_US-Cop.nc --all-times

# Convert specific time steps with custom output
python nc_to_image.py landsat_landsat_US-Cop.nc -t 0 10 20 30 -o my_images

# Convert to JPEG format
python nc_to_image.py landsat_landsat_US-Cop.nc -f jpeg --all-times
```

## Output Structure

```
output_folder/
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

## Common Use Cases

### Grassland Monitoring
```bash
# Extract key bands for vegetation analysis
python nc_to_image.py landsat_landsat_US-Cop.nc --all-times -o grassland_analysis

# Focus on specific years (time indices vary by dataset)
python nc_to_image.py landsat_landsat_US-Cop.nc -t 100 200 300 400 500
```

### Quick Visual Assessment
```bash
# Just RGB and false-color composites
python nc_to_image.py landsat_landsat_US-Cop.nc -t 0 50 100 150 200
```

### AlphaEarth Embeddings
```bash
# Extract embedding visualizations
python nc_to_image.py embeddings_alpha_earth_US-Cop.nc --all-times -o embeddings_viz
```

## File Naming Convention

Images are named with actual acquisition dates:
- `landsat_landsat_US-Cop_rgb_2002-01-12.png`
- `landsat_landsat_US-Cop_nir08_2024-09-29.png`
- `embeddings_alpha_earth_US-Cop_embedding_b00_2017-01-01.png`

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
                        Output directory (default: ./output_images)
  -f {png,jpg,jpeg,tiff}, --format {png,jpg,jpeg,tiff}
                        Output image format (default: png)
  -t TIME_INDICES [TIME_INDICES ...]
                        Time indices to process (default: first time step only)
  --all-times           Process all time steps (WARNING: can create thousands of images!)
  --no-composites       Skip creating RGB/false-color composites
```

## Data Overview

- **Landsat file**: 778 time steps (2002-2024), 14 bands → ~10,892 images with `--all-times`
- **AlphaEarth file**: 8 time steps (2017-2024), 5 embedding bands → ~40 images with `--all-times`

## Image Interpolation

Use `interpolate_images.py` to generate missing images between two dates using AI:

```bash
# Set up your Gemini API key in .env file first
# Get free key from: https://aistudio.google.com/app/apikey

# Binary recursive interpolation (default, recommended)
python interpolate_images.py image_2002-01-12.png image_2002-03-01.png

# Binary interpolation generates middle first, then 1/4, 3/4, etc.
# Order: middle → quarters → eighths → sixteenths...
# More balanced and intelligent than linear day-by-day

# Linear sequential interpolation (day by day)
python interpolate_images.py start.png end.png --linear

# Simple interpolation without AI (fastest)
python interpolate_images.py start.png end.png --simple-only --binary
```

### Interpolation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Binary (default)** | Middle first, then recursive halves | AI-guided, balanced progression |
| **Linear** | Day-by-day sequential | Simple temporal progression |

**Example**: Between Jan 12 and Mar 1 (48 days apart), it generates 47 intermediate images for every missing day!

## Requirements

- Python packages: `xarray`, `netcdf4`, `matplotlib`, `pillow`, `numpy`, `pandas`, `google-generativeai`, `python-dotenv`
- Already installed in the `venv/` environment
- Optional: Gemini API key for AI-powered interpolation