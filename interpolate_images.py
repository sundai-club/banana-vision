#!/usr/bin/env python3
"""
Satellite Image Interpolation using Gemini 2.5 Flash Image Preview

This script uses Google's Gemini 2.5 Flash Image Preview model to generate realistic
intermediate satellite images between two known images with timestamps. The model
can create sophisticated interpolations considering seasonal changes, weather patterns,
and agricultural cycles. Useful for filling gaps in time series satellite data.

Set up your API key in .env file: GEMINI_API_KEY=your_key_here
"""

import argparse
import os
import sys
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import base64
from io import BytesIO

import google.generativeai as genai
from google import genai as google_genai
from PIL import Image
import requests
from dotenv import load_dotenv


def extract_date_from_filename(filename):
    """Extract date from filename with format: *_YYYY-MM-DD.ext"""
    pattern = r'(\d{4}-\d{2}-\d{2})'
    match = re.search(pattern, filename)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d')
    return None




def generate_binary_interpolation_sequence(start_date, end_date):
    """
    Generate dates for binary recursive interpolation.
    Returns list of (target_date, reference_start, reference_end) tuples.
    """
    def binary_split(left_date, right_date, depth=0, max_depth=10):
        """Recursively split date range in binary fashion."""
        if depth >= max_depth or (right_date - left_date).days <= 1:
            return []
        
        # Calculate middle date
        total_days = (right_date - left_date).days
        middle_days = total_days // 2
        middle_date = left_date + timedelta(days=middle_days)
        
        # Add middle point first
        splits = [(middle_date, left_date, right_date)]
        
        # Recursively split left and right halves
        splits.extend(binary_split(left_date, middle_date, depth + 1, max_depth))
        splits.extend(binary_split(middle_date, right_date, depth + 1, max_depth))
        
        return splits
    
    return binary_split(start_date, end_date)


def encode_image_to_base64(image_path):
    """Encode image to base64 for Gemini API."""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def create_interpolation_prompt(start_date, end_date, target_date, image_type="satellite"):
    """Create focused prompt for Gemini to generate interpolated images."""
    
    days_total = (end_date - start_date).days
    days_from_start = (target_date - start_date).days
    interpolation_ratio = days_from_start / days_total
    
    prompt = f"""Create an image that interpolates between these two satellite images for date {target_date.strftime('%Y-%m-%d')}.

The interpolation should be at position {interpolation_ratio:.2f} in the temporal sequence.

Output: Generate a new satellite image (not text description) that smoothly transitions between the two input images."""
    return prompt


def interpolate_with_gemini(api_key, start_image_path, end_image_path, target_date, output_path, image_type="satellite"):
    """Use Gemini 2.5 Flash Image Preview to generate interpolated satellite images."""
    
    # Extract dates from filenames
    start_date = extract_date_from_filename(start_image_path.name)
    end_date = extract_date_from_filename(end_image_path.name)
    
    if not start_date or not end_date:
        raise ValueError("Could not extract dates from filenames")
    
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    
    if target_date <= start_date or target_date >= end_date:
        raise ValueError("Target date must be between start and end dates")
    
    print(f"Interpolating between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
    print(f"Target date: {target_date.strftime('%Y-%m-%d')}")
    
    # Load images
    start_img = Image.open(start_image_path)
    end_img = Image.open(end_image_path)
    
    # Create prompt
    prompt = create_interpolation_prompt(start_date, end_date, target_date, image_type)
    
    try:
        # Try with newer Google GenAI client first
        print("Trying Google GenAI client...")
        client = google_genai.Client(api_key=api_key)
        
        # Create the chat
        chat = client.chats.create(model="gemini-2.5-flash-image-preview")
        
        # Send the image and ask for it to be edited
        response = chat.send_message([prompt, start_img, end_img])
        
        # Get the text and the image generated
        generated_image_saved = False
        for i, part in enumerate(response.candidates[0].content.parts):
            if part.text is not None:
                print("Response from Gemini:")
                print(part.text)
                
            elif part.inline_data is not None:
                # Extract and save the generated image
                image_bytes = part.inline_data.data
                generated_image = Image.open(BytesIO(image_bytes))
                generated_image.save(output_path)
                print(f"Generated image saved: {output_path}")
                generated_image_saved = True
        
        if generated_image_saved:
            return output_path
        else:
            print("Warning: No image was generated by Gemini")
            return create_simple_interpolation(start_img, end_img, start_date, end_date, target_date, output_path)
        
    except Exception as e:
        print(f"Error with Google GenAI client: {e}")
        print("Trying legacy google.generativeai client...")
        
        try:
            # Fallback to legacy client
            genai.configure(api_key=api_key)
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-image-preview')
            model = genai.GenerativeModel(model_name)
            print(f"Using Gemini model: {model_name}")
            
            # Generate interpolated image using chat mode
            chat = model.start_chat()
            response = chat.send_message([prompt, start_img, end_img])
            
            # Check for generated images in the response
            generated_image_saved = False
            
            for i, part in enumerate(response.candidates[0].content.parts):
                if part.text is not None:
                    print("Response from Gemini:")
                    print(part.text)
                    
                elif part.inline_data is not None:
                    # Extract and save the generated image
                    image_bytes = part.inline_data.data
                    generated_image = Image.open(BytesIO(image_bytes))
                    generated_image.save(output_path)
                    print(f"Generated image saved: {output_path}")
                    generated_image_saved = True
            
            if generated_image_saved:
                return output_path
            else:
                print("Warning: No image was generated by Gemini")
                return create_simple_interpolation(start_img, end_img, start_date, end_date, target_date, output_path)
                
        except Exception as e2:
            print(f"Error with legacy client: {e2}")
            print("Falling back to simple interpolation...")
            return create_simple_interpolation(start_img, end_img, start_date, end_date, target_date, output_path)


def create_simple_interpolation(start_img, end_img, start_date, end_date, target_date, output_path):
    """Create a simple linear interpolation between two images as fallback."""
    
    import numpy as np
    
    # Calculate interpolation weight
    days_total = (end_date - start_date).days
    days_from_start = (target_date - start_date).days
    weight = days_from_start / days_total
    
    print(f"Creating simple interpolation with weight {weight:.3f}")
    
    # Convert to numpy arrays
    start_array = np.array(start_img)
    end_array = np.array(end_img)
    
    # Ensure same shape
    if start_array.shape != end_array.shape:
        print(f"Warning: Image shapes don't match: {start_array.shape} vs {end_array.shape}")
        return None
    
    # Linear interpolation
    interpolated = (1 - weight) * start_array + weight * end_array
    interpolated = interpolated.astype(np.uint8)
    
    # Convert back to PIL Image
    result_img = Image.fromarray(interpolated)
    
    # Save interpolated image
    result_img.save(output_path)
    print(f"Saved interpolated image: {output_path}")
    
    return output_path


def copy_original_images(start_image_path, end_image_path, output_dir):
    """Copy the original start and end images to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy start image
    start_path = Path(start_image_path)
    start_dest = output_path / start_path.name
    shutil.copy2(start_image_path, start_dest)
    print(f"Copied original start image: {start_dest}")
    
    # Copy end image  
    end_path = Path(end_image_path)
    end_dest = output_path / end_path.name
    shutil.copy2(end_image_path, end_dest)
    print(f"Copied original end image: {end_dest}")


def interpolate_binary_recursive(start_image_path, end_image_path, output_dir, api_key=None, image_type="satellite"):
    """Generate images using binary recursive interpolation."""
    
    start_path = Path(start_image_path)
    end_path = Path(end_image_path)
    output_path = Path(output_dir)
    
    # Extract dates
    start_date = extract_date_from_filename(start_path.name)
    end_date = extract_date_from_filename(end_path.name)
    
    if not start_date or not end_date:
        print("Error: Could not extract dates from filenames")
        return
    
    # Create output directory and copy original images
    output_path.mkdir(parents=True, exist_ok=True)
    copy_original_images(start_image_path, end_image_path, output_dir)
    
    # Get binary interpolation sequence
    interpolation_sequence = generate_binary_interpolation_sequence(start_date, end_date)
    
    if not interpolation_sequence:
        print("No missing dates between the two images")
        return
    
    print(f"Binary interpolation: {len(interpolation_sequence)} images to generate")
    print(f"Sequence order: {[item[0].strftime('%Y-%m-%d') for item in interpolation_sequence[:5]]}...")
    
    # Determine band/channel from filename
    band_match = re.search(r'_(rgb|red|green|blue|nir08|false_color|embedding)_', start_path.name)
    band_type = band_match.group(1) if band_match else "unknown"
    
    # Keep track of generated images for future reference
    generated_images = {
        start_date: start_path,
        end_date: end_path
    }
    
    # Process interpolation sequence
    for target_date, ref_start_date, ref_end_date in interpolation_sequence:
        print(f"\nGenerating {target_date.strftime('%Y-%m-%d')} between {ref_start_date.strftime('%Y-%m-%d')} and {ref_end_date.strftime('%Y-%m-%d')}")
        
        # Find the best reference images (could be original or previously generated)
        ref_start_path = generated_images.get(ref_start_date, start_path)
        ref_end_path = generated_images.get(ref_end_date, end_path)
        
        # Create output filename
        base_name = re.sub(r'_\d{4}-\d{2}-\d{2}\.', f'_{target_date.strftime("%Y-%m-%d")}.', start_path.name)
        output_file = output_path / base_name
        
        # Generate interpolated image
        if api_key:
            try:
                result_path = interpolate_with_gemini(api_key, ref_start_path, ref_end_path, target_date, 
                                      output_file, f"{image_type} {band_type}")
            except Exception as e:
                print(f"Error with Gemini interpolation: {e}")
                result_path = create_simple_interpolation(
                    Image.open(ref_start_path), Image.open(ref_end_path),
                    ref_start_date, ref_end_date, target_date, output_file
                )
        else:
            result_path = create_simple_interpolation(
                Image.open(ref_start_path), Image.open(ref_end_path),
                ref_start_date, ref_end_date, target_date, output_file
            )
        
        # Add generated image to reference pool
        if result_path:
            generated_images[target_date] = result_path



def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Interpolate satellite images using Gemini 2.5 Flash')
    parser.add_argument('start_image', help='First satellite image with date in filename')
    parser.add_argument('end_image', help='Second satellite image with date in filename')
    parser.add_argument('-o', '--output', default='./output/interpolated_images',
                       help='Output directory for interpolated images')
    parser.add_argument('--api-key', help='Google Gemini API key (or set in .env file as GEMINI_API_KEY)')
    parser.add_argument('--image-type', default='satellite',
                       help='Type of imagery (satellite, rgb, false_color, etc.)')
    parser.add_argument('--simple-only', action='store_true',
                       help='Skip Gemini image generation and use simple linear interpolation as fallback')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    
    if not args.simple_only and not api_key:
        print("Warning: No Gemini API key provided. Using simple interpolation as fallback.")
        print("Set GEMINI_API_KEY environment variable or use --api-key option to enable AI-powered interpolation.")
        api_key = None
    
    # Validate input files
    if not Path(args.start_image).exists():
        print(f"Error: Start image not found: {args.start_image}")
        sys.exit(1)
    
    if not Path(args.end_image).exists():
        print(f"Error: End image not found: {args.end_image}")
        sys.exit(1)
    
    # Run binary recursive interpolation
    if api_key and not args.simple_only:
        print("Using Gemini AI-powered binary recursive interpolation")
    else:
        print("Using simple mathematical binary recursive interpolation")
    
    interpolate_binary_recursive(
        args.start_image, 
        args.end_image, 
        args.output,
        api_key if not args.simple_only else None,
        args.image_type
    )
    
    print("\nInterpolation complete!")


if __name__ == '__main__':
    main()