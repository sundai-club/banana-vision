Step 1: download files like `landsat_landsat_US-Cop.nc` and `embeddings_alpha_earth_US-Cop.nc` and put them into `input` subfolder

Step 2: extract images from all NC files to `output` folder:
```
python nc_to_image.py input --all-times
```

Step 3: interpolate any particular 2 images using gemini nano banana image model
```
python interpolate_images.py output/rgb/landsat_landsat_US-Cop_x641190_y4217025_rgb_2002-10-19.png output/rgb/landsat_landsat_US-Cop_x641190_y4217025_rgb_2002-11-12.png 
```