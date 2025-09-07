#!/usr/bin/env python3
"""
NetCDF to Image Converter

Converts NetCDF files containing satellite reflectance data or embeddings to images.
Supports RGB composites for reflectance data and individual band/variable outputs.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


def normalize_data(data, percentiles=(2, 98)):
    """Normalize data to 0-1 range using percentile clipping."""
    valid_data = data[~np.isnan(data)]
    if valid_data.size == 0:
        return np.zeros_like(data)
    p_low, p_high = np.percentile(valid_data, percentiles)
    data_clipped = np.clip(data, p_low, p_high)
    return (data_clipped - p_low) / (p_high - p_low)


def create_quality_mask(ds, time_idx, quality_threshold='medium'):
    """Create a quality mask for valid data based on QA flags."""
    qa_pixel = ds.get('qa_pixel', None)
    if qa_pixel is None:
        # No QA data available, mask only zeros
        red = ds['red'].isel(time=time_idx).values if 'red' in ds.data_vars else None
        if red is not None:
            return red > 0
        return np.ones(ds.dims['y'], ds.dims['x'], dtype=bool)
    
    qa_data = qa_pixel.isel(time=time_idx).values
    
    # Define quality thresholds based on Landsat QA_PIXEL flags
    # These are common good quality flags
    if quality_threshold == 'high':
        # Only clear land pixels
        good_qa = [20, 24]  # Clear land, no clouds/shadows
    elif quality_threshold == 'medium':
        # Clear land and some clear water
        good_qa = [20, 24, 32, 36, 40]  # Includes some clear conditions
    else:  # 'low' - more permissive
        # Include more conditions but exclude obvious bad data
        bad_qa = [72, 73, 74, 75, 104, 105, 106, 107, 112, 113, 114, 115]  # Clouds, shadows
        return ~np.isin(qa_data, bad_qa)
    
    return np.isin(qa_data, good_qa)


def normalize_data_with_quality(data, quality_mask, percentiles=(2, 98)):
    """Normalize data using only high-quality pixels for statistics."""
    valid_mask = quality_mask & (data > 0) & (~np.isnan(data))
    
    if valid_mask.sum() == 0:
        return np.zeros_like(data), np.zeros_like(data, dtype=bool)
    
    valid_data = data[valid_mask]
    p_low, p_high = np.percentile(valid_data, percentiles)
    
    # Normalize only valid pixels
    normalized = np.zeros_like(data)
    normalized[valid_mask] = np.clip(
        (data[valid_mask] - p_low) / (p_high - p_low), 0, 1
    )
    
    return normalized, valid_mask


def create_rgb_composite(ds, time_idx=0, rgb_bands=('red', 'green', 'blue')):
    """Create RGB composite from reflectance bands."""
    try:
        # Extract RGB bands for specific time
        red = ds[rgb_bands[0]].isel(time=time_idx).values
        green = ds[rgb_bands[1]].isel(time=time_idx).values
        blue = ds[rgb_bands[2]].isel(time=time_idx).values
        
        # Normalize each band
        red_norm = normalize_data(red)
        green_norm = normalize_data(green)
        blue_norm = normalize_data(blue)
        
        # Stack into RGB image
        rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
        
        # Convert to 8-bit
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image
        
    except KeyError as e:
        print(f"Warning: RGB bands not available: {e}")
        return None


def create_false_color_composite(ds, time_idx=0, bands=('nir08', 'red', 'green')):
    """Create false color composite (NIR-Red-Green)."""
    try:
        # Extract bands for specific time
        nir = ds[bands[0]].isel(time=time_idx).values
        red = ds[bands[1]].isel(time=time_idx).values
        green = ds[bands[2]].isel(time=time_idx).values
        
        # Normalize each band
        nir_norm = normalize_data(nir)
        red_norm = normalize_data(red)
        green_norm = normalize_data(green)
        
        # Stack into false color image
        composite = np.stack([nir_norm, red_norm, green_norm], axis=2)
        
        # Convert to 8-bit
        composite = (composite * 255).astype(np.uint8)
        
        return composite
        
    except KeyError as e:
        print(f"Warning: False color bands not available: {e}")
        return None


def save_single_band(ds, variable, time_idx, output_path, format='png'):
    """Save a single variable/band as grayscale image."""
    data = ds[variable].isel(time=time_idx).values
    
    # Handle different dimensionalities
    if data.ndim > 2:
        print(f"Warning: {variable} has {data.ndim} dimensions, using first slice")
        data = data[0] if data.ndim == 3 else data[0, 0]
    
    # Normalize data
    data_norm = normalize_data(data)
    
    # Convert to 8-bit
    data_8bit = (data_norm * 255).astype(np.uint8)
    
    # Save using PIL
    img = Image.fromarray(data_8bit, mode='L')
    img.save(output_path, format=format.upper())


def save_embedding_visualization(ds, time_idx, band_idx, output_path, format='png'):
    """Save embedding band as grayscale image."""
    data = ds['embedding'].isel(time=time_idx, band=band_idx).values
    
    # Normalize data
    data_norm = normalize_data(data)
    
    # Convert to 8-bit
    data_8bit = (data_norm * 255).astype(np.uint8)
    
    # Save using PIL
    img = Image.fromarray(data_8bit, mode='L')
    img.save(output_path, format=format.upper())


def create_upscaled_image(image_array, scale_factor=4, interpolation='bilinear'):
    """Upscale image using PIL interpolation methods."""
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
        mode = 'RGBA'
    elif len(image_array.shape) == 3 and image_array.shape[2] == 2:  # LA
        mode = 'LA'
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
        mode = 'RGB'
    else:  # Grayscale
        mode = 'L'
    
    # Convert to PIL image
    img = Image.fromarray(image_array, mode=mode)
    
    # Set interpolation method
    interpolation_methods = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR, 
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    interp = interpolation_methods.get(interpolation, Image.BILINEAR)
    
    # Calculate new size
    new_width = img.width * scale_factor
    new_height = img.height * scale_factor
    
    # Upscale
    upscaled = img.resize((new_width, new_height), interp)
    
    return np.array(upscaled)


def create_temporal_composite(ds, time_indices, method='median'):
    """Create composite image from multiple time steps."""
    rgb_bands = ['red', 'green', 'blue']
    
    if not all(band in ds.data_vars for band in rgb_bands):
        return None
    
    print(f"Creating temporal composite from {len(time_indices)} time steps using {method} method")
    
    # Collect data from all time steps
    all_red = []
    all_green = []
    all_blue = []
    
    for time_idx in time_indices:
        # Get band data
        red = ds['red'].isel(time=time_idx).values
        green = ds['green'].isel(time=time_idx).values  
        blue = ds['blue'].isel(time=time_idx).values
        
        # Normalize each band
        red_norm = normalize_data(red)
        green_norm = normalize_data(green)
        blue_norm = normalize_data(blue)
        
        all_red.append(red_norm)
        all_green.append(green_norm)
        all_blue.append(blue_norm)
    
    # Stack arrays
    red_stack = np.stack(all_red, axis=2)  # shape: (y, x, time)
    green_stack = np.stack(all_green, axis=2)
    blue_stack = np.stack(all_blue, axis=2)
    
    # Create composite based on method
    if method == 'median':
        red_comp = np.median(red_stack, axis=2)
        green_comp = np.median(green_stack, axis=2)
        blue_comp = np.median(blue_stack, axis=2)
    elif method == 'mean':
        red_comp = np.mean(red_stack, axis=2)
        green_comp = np.mean(green_stack, axis=2)
        blue_comp = np.mean(blue_stack, axis=2)
    else:  # 'max'
        red_comp = np.max(red_stack, axis=2)
        green_comp = np.max(green_stack, axis=2)  
        blue_comp = np.max(blue_stack, axis=2)
    
    # Create RGB image
    composite_rgb = np.stack([red_comp, green_comp, blue_comp], axis=2)
    composite_rgb = (composite_rgb * 255).astype(np.uint8)
    
    return composite_rgb


def create_padded_image(image_array, pad_size=2, pad_color=(64, 64, 64, 255)):
    """Add padding around image to make it larger."""
    if len(image_array.shape) == 3:
        height, width, channels = image_array.shape
        # Create larger array filled with pad_color
        padded = np.full((height * (1 + 2*pad_size), width * (1 + 2*pad_size), channels), 
                        pad_color[:channels], dtype=image_array.dtype)
        
        # Place original image in center
        y_start = height * pad_size
        x_start = width * pad_size  
        padded[y_start:y_start+height, x_start:x_start+width] = image_array
        
    else:  # Grayscale
        height, width = image_array.shape
        padded = np.full((height * (1 + 2*pad_size), width * (1 + 2*pad_size)), 
                        pad_color[0], dtype=image_array.dtype)
        
        y_start = height * pad_size
        x_start = width * pad_size
        padded[y_start:y_start+height, x_start:x_start+width] = image_array
    
    return padded


def save_single_band_array(ds, variable, time_idx, quality_threshold='medium'):
    """Return single band as array instead of saving directly."""
    data = ds[variable].isel(time=time_idx).values
    
    # Handle different dimensionalities
    if data.ndim > 2:
        data = data[0] if data.ndim == 3 else data[0, 0]
    
    # Create quality mask
    quality_mask = create_quality_mask(ds, time_idx, quality_threshold)
    
    # Normalize data using quality mask
    data_norm, valid_mask = normalize_data_with_quality(data, quality_mask)
    
    # Create grayscale + alpha image
    img_array = np.zeros((*data.shape, 2), dtype=np.uint8)
    img_array[:, :, 0] = (data_norm * 255).astype(np.uint8)  # Grayscale
    img_array[:, :, 1] = np.where(valid_mask, 255, 0)  # Alpha
    
    return img_array


def apply_image_enhancements(image_array, scale_factor=1, pad_size=0, interpolation='bilinear'):
    """Apply scaling and padding enhancements."""
    enhanced = image_array
    
    # Apply scaling first
    if scale_factor > 1:
        enhanced = create_upscaled_image(enhanced, scale_factor, interpolation)
    
    # Apply padding
    if pad_size > 0:
        enhanced = create_padded_image(enhanced, pad_size)
    
    return enhanced


def extract_coordinate_bounds(ds):
    """Extract coordinate bounds from NetCDF dataset for filename."""
    try:
        # Get coordinate bounds
        if 'x' in ds.coords and 'y' in ds.coords:
            x_min, x_max = float(ds.x.min()), float(ds.x.max())
            y_min, y_max = float(ds.y.min()), float(ds.y.max())
            
            # Format coordinates for filename (remove decimals, use shorter format)
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            
            # Create coordinate string for filename
            coord_str = f"x{x_center}_y{y_center}"
            return coord_str
        else:
            return "nocoords"
    except Exception as e:
        print(f"Warning: Could not extract coordinates: {e}")
        return "nocoords"


def save_enhanced_image(image_array, output_path, format='png'):
    """Save enhanced image array."""
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
        mode = 'RGBA'
    elif len(image_array.shape) == 3 and image_array.shape[2] == 2:  # LA
        mode = 'LA'
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
        mode = 'RGB'
    else:  # Grayscale
        mode = 'L'
    
    img = Image.fromarray(image_array, mode=mode)
    img.save(output_path, format=format.upper())


def process_netcdf_file(input_file, output_dir, format='png', time_indices=None, create_composites=True, all_times=False, scale_factor=1, temporal_composite=None, composite_method='median', pad_size=0, interpolation='bilinear'):
    """Process a NetCDF file and convert to images."""
    
    print(f"Processing {input_file}...")
    
    # Open dataset
    ds = xr.open_dataset(input_file)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get base filename and coordinate info
    base_name = Path(input_file).stem
    coord_str = extract_coordinate_bounds(ds)
    
    # Determine time indices to process
    num_times = ds.dims.get('time', 1)
    if all_times:
        time_indices = list(range(num_times))
        print(f"Processing ALL {num_times} time steps - this will create {num_times * 14} images!")
    elif time_indices is None:
        time_indices = [0] if num_times > 0 else [0]
        if num_times > 1:
            print(f"Note: Processing only first time step. File has {num_times} time steps.")
            print(f"Use --all-times to process all {num_times} time steps.")
    
    time_indices = [t for t in time_indices if t < num_times]
    
    for time_idx in time_indices:
        # Get actual date/time for filename
        if 'time' in ds.coords and num_times > 1:
            time_value = ds['time'].isel(time=time_idx).values
            # Convert numpy datetime64 to string format YYYY-MM-DD
            date_str = pd.to_datetime(time_value).strftime('%Y-%m-%d')
            time_suffix = f"_{date_str}"
        else:
            time_suffix = f"_t{time_idx:03d}" if num_times > 1 else ""
        
        # Check if this looks like reflectance data (Landsat/MODIS)
        reflectance_bands = ['red', 'green', 'blue', 'nir08']
        has_reflectance = all(band in ds.data_vars for band in reflectance_bands[:3])
        
        if has_reflectance and create_composites:
            # Create RGB composite with enhancements
            rgb_img = create_rgb_composite(ds, time_idx)
            if rgb_img is not None:
                rgb_img = apply_image_enhancements(rgb_img, scale_factor, pad_size, interpolation)
                rgb_dir = output_path / "rgb"
                rgb_dir.mkdir(exist_ok=True)
                rgb_path = rgb_dir / f"{base_name}_{coord_str}_rgb{time_suffix}.{format}"
                save_enhanced_image(rgb_img, rgb_path, format)
                print(f"Saved RGB composite: {rgb_path}")
            
            # Create false color composite if NIR available
            if 'nir08' in ds.data_vars:
                false_color_img = create_false_color_composite(ds, time_idx)
                if false_color_img is not None:
                    false_color_img = apply_image_enhancements(false_color_img, scale_factor, pad_size, interpolation)
                    fc_dir = output_path / "false_color"
                    fc_dir.mkdir(exist_ok=True)
                    fc_path = fc_dir / f"{base_name}_{coord_str}_false_color{time_suffix}.{format}"
                    save_enhanced_image(false_color_img, fc_path, format)
                    print(f"Saved false color composite: {fc_path}")
        
        # Save individual bands/variables
        for var_name in ds.data_vars:
            var_data = ds[var_name]
            
            # Skip if variable doesn't have spatial dimensions
            if 'y' not in var_data.dims or 'x' not in var_data.dims:
                continue
                
            # Handle embeddings specially
            if var_name == 'embedding' and 'band' in var_data.dims:
                # Save first few embedding bands
                emb_dir = output_path / "embeddings"
                emb_dir.mkdir(exist_ok=True)
                num_bands = min(5, var_data.sizes['band'])
                for band_idx in range(num_bands):
                    emb_path = emb_dir / f"{base_name}_{coord_str}_embedding_b{band_idx:02d}{time_suffix}.{format}"
                    save_embedding_visualization(ds, time_idx, band_idx, emb_path, format)
                    print(f"Saved embedding band {band_idx}: {emb_path}")
            else:
                # Create subdirectory for this band/variable
                var_dir = output_path / var_name
                var_dir.mkdir(exist_ok=True)
                var_path = var_dir / f"{base_name}_{coord_str}_{var_name}{time_suffix}.{format}"
                save_single_band(ds, var_name, time_idx, var_path, format)
                print(f"Saved {var_name}: {var_path}")
    
    # Create temporal composite if requested
    if temporal_composite is not None and has_reflectance:
        print(f"\nCreating temporal composite...")
        composite_img = create_temporal_composite(ds, temporal_composite, composite_method)
        if composite_img is not None:
            # Apply enhancements
            composite_img = apply_image_enhancements(composite_img, scale_factor, pad_size, interpolation)
            
            # Save temporal composite
            composite_dir = output_path / "temporal_composite"  
            composite_dir.mkdir(exist_ok=True)
            composite_path = composite_dir / f"{base_name}_{coord_str}_temporal_{composite_method}_{len(temporal_composite)}imgs.{format}"
            
            img = Image.fromarray(composite_img, 'RGBA')
            img.save(composite_path, format=format.upper())
            print(f"Saved temporal composite: {composite_path}")
    
    ds.close()
    print(f"Completed processing {input_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert NetCDF files to images')
    parser.add_argument('input', help='Input NetCDF file or directory')
    parser.add_argument('-o', '--output', default='./output',
                       help='Output directory (default: ./output)')
    parser.add_argument('-f', '--format', choices=['png', 'jpg', 'jpeg', 'tiff'], 
                       default='png', help='Output image format (default: png)')
    parser.add_argument('-t', '--time-indices', type=int, nargs='+',
                       help='Time indices to process (default: first time step only)')
    parser.add_argument('--all-times', action='store_true',
                       help='Process all time steps (WARNING: can create thousands of images!)')
    parser.add_argument('--no-composites', action='store_true',
                       help='Skip creating RGB/false-color composites')
    parser.add_argument('--scale', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 8, 10],
                       help='Upscale factor for larger images (default: 1, no scaling)')
    parser.add_argument('--temporal-composite', type=int, nargs='+', 
                       help='Create temporal composite from multiple time indices')
    parser.add_argument('--composite-method', choices=['median', 'mean', 'max'], default='median',
                       help='Method for temporal compositing (default: median)')
    parser.add_argument('--pad', type=int, default=0,
                       help='Add padding around images (multiplier of original size)')
    parser.add_argument('--interpolation', choices=['nearest', 'bilinear', 'bicubic', 'lanczos'], 
                       default='bilinear', help='Interpolation method for upscaling')
    
    args = parser.parse_args()
    
    # Handle input path
    input_path = Path(args.input)
    
    if input_path.is_file():
        if input_path.suffix == '.nc':
            process_netcdf_file(str(input_path), args.output, args.format, 
                              args.time_indices, not args.no_composites, args.all_times,
                              args.scale, args.temporal_composite, args.composite_method, args.pad, args.interpolation)
        else:
            print(f"Error: {input_path} is not a NetCDF file (.nc)")
            sys.exit(1)
    elif input_path.is_dir():
        # Process all .nc files in directory
        nc_files = list(input_path.glob('*.nc'))
        if not nc_files:
            print(f"No NetCDF files found in {input_path}")
            sys.exit(1)
        
        for nc_file in nc_files:
            try:
                process_netcdf_file(str(nc_file), args.output, args.format,
                                  args.time_indices, not args.no_composites, args.all_times,
                                  args.scale, args.temporal_composite, args.composite_method, args.pad, args.interpolation)
            except Exception as e:
                print(f"Error processing {nc_file}: {e}")
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)
    
    print("\nConversion complete!")


if __name__ == '__main__':
    main()