# step 1. put landsat_landsat_US-Cop.nc and embeddings_alpha_earth_US-Cop.nc to input folder
# step 2: extract NC to output dir
python nc_to_image.py input --all-times
# step 3: interpolate 2 images using gemini nano banana image model
python interpolate_images.py output/rgb/landsat_landsat_US-Cop_x641190_y4217025_rgb_2002-10-19.png output/rgb/landsat_landsat_US-Cop_x641190_y4217025_rgb_2002-11-12.png 