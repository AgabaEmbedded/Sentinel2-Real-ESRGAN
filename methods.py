import os
import glob
import shutil
import rasterio
import numpy as np
from PIL import Image
def update_metadata(metadata, scale_factor, original_transform, original_res, height):
      
  new_transform = original_transform * original_transform.scale(1/scale_factor, 1/scale_factor)
  new_res = (original_res[0] / scale_factor, original_res[1] / scale_factor)


  # Update metadata for the new resolution
  metadata.update({
      "height": height*scale_factor,  
      "width": height*scale_factor,  
      "transform": new_transform,    
      "dtype": "uint16"  # Adjust if necessary
  })

  return metadata
def extract_patches(image_folder, patch_size, save_directory, scale_factor):
  for image_dir in sorted(os.listdir(image_folder)):

    image_path = os.path.join(image_folder, image_dir)

    with rasterio.open(image_path) as tiff_image:
      metadata = tiff_image.meta.copy()
      original_transform = tiff_image.transform
      original_res = tiff_image.res
      image = tiff_image.read(1)

      h, w = image.shape
      image_name = image_dir.replace(".tif", "")
      metadata_dict[image_name]=update_metadata(metadata, scale_factor, original_transform, original_res, h)



    for i in range(0, h - patch_size + 1, patch_size):
      for j in range(0, w - patch_size + 1, patch_size):
        patch = image[i:i+patch_size, j:j+patch_size]

        patch = Image.fromarray(patch)
        save_path = os.path.join(save_directory, f"{image_name}_{h}_{i}_{j}.png")
        patch.save(save_path)

    print(f"done patching image {image_name}")


def reconstruct_image(patch_folder, scaling_factor, output_path):
  # Iterate over patches
  for patch_file in sorted(os.listdir(patch_folder)):
    if patch_file.endswith('.png'):
      # Extract shape, row and column indices from filename
      parts = patch_file.rsplit('_', 4)
      print(parts)
      width = height = int(parts[-4])*scaling_factor
      
      i = int(parts[-3])*scaling_factor
      j = int(parts[-2])*scaling_factor
      image_name = parts[0]

      # Place the patch in the correct position
      if i == 0 and j == 0:
        reconstructed_image = np.zeros((height, width), dtype=np.float32)
        count = 1
        
      # Load the patch
      patch_path = os.path.join(patch_folder, patch_file)
      patch = np.array(Image.open(patch_path))
      patch_size = patch.shape[0]

      # Place the patch in the correct position
      print(patch.shape)
      print(f"{i}, {i+patch_size},{j}, {j+patch_size}")
      print(reconstructed_image.shape)
      print(f"{height}, {width}")
      reconstructed_image[i:i+patch_size, j:j+patch_size] = patch

      if (height/patch_size)**2 == count:
        # Convert to image and save
        metadata = metadata_dict[image_name]
        output_path = os.path.join(outut_path, f"{image_name}_out.tif")
        print(output_path)
        with rasterio.open("results/abc.tif", "w", **metadata) as dst:
          dst.write(reconstructed_image[np.newaxis, :, :])  # Add a band dimension

          count+=1
# utils for visualization
def display(img1, img2):
  fig = plt.figure(figsize=(25, 10))
  ax1 = fig.add_subplot(1, 2, 1)
  plt.title('Input image', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  plt.title('Real-ESRGAN output', fontsize=16)
  ax2.axis('off')
  ax1.imshow(img1, cmap = "gray")
  ax2.imshow(img2, cmap = "gray")