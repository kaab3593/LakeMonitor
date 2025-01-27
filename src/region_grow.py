
import numpy as np
from skimage import exposure 
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import glob2
from skimage import io, color
from skimage.util import img_as_ubyte
import sys
import os

class RegionGrow:
    """Region growing algorithm from an initial seed determined as the centroid 
       of the largest connected component in a binarized image."""

    def __init__(self, in_file, out_file):
        self.in_file = in_file
        self.out_file = out_file
        
        self.image = self.load_image()
        # estimate a seed point from binary image
        self.seed_xy = self.seed(0.1)

        self.image = self.region_growing(self.image, self.seed_xy)
        self.save_image()

    def load_image(self):
        im = io.imread(self.in_file)
        return im
        
    def seed(self, thresh):
        gray_image = color.rgb2gray(self.image) 
        gray_image = exposure.equalize_hist(gray_image)
        binary_mask = gray_image < thresh

        # Label the connected components in the binary mask
        labeled_mask = label(binary_mask)
        # Get properties of the labeled regions
        regions = regionprops(labeled_mask)
        # Find the largest region by area
        largest_region = max(regions, key=lambda region: region.area)
        # Create a binary mask for the largest component
        largest_component_mask = labeled_mask == largest_region.label
        # Segment the largest object
        segmented_largest_component = np.zeros_like(gray_image)
        segmented_largest_component[largest_component_mask] = gray_image[largest_component_mask]
        # Compute the distance transform
        dist_transform = distance_transform_edt(segmented_largest_component)
        # Find the seed point (pixel with maximum distance)
        seed_point = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        y, x = seed_point

        return (y,x)


    def region_growing(self, image, seed, threshold=0.05):
        """ Perform region growing segmentation. """
        # Convert image to grayscale
        gray_image = color.rgb2gray(image)
        gray_image = exposure.equalize_hist(gray_image)
        # Initialize the binary mask
        binary_mask = np.zeros_like(gray_image, dtype=bool)
        # Get the intensity value of the seed point
        seed_value = gray_image[seed]
        # Create a list of pixels to examine, starting with the seed
        pixels_to_examine = [seed]
        
        # Grow from the seed position
        while pixels_to_examine:
            # Get the current pixel to examine
            current_pixel = pixels_to_examine.pop()
            x, y = current_pixel
            # Skip if the pixel is already in the region
            if binary_mask[x, y]:
                continue
            # Add pixel to region if within the threshold
            if abs(gray_image[x, y] - seed_value) < threshold:
                binary_mask[x, y] = True
                # Add neighboring pixels to the list
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gray_image.shape[0] and 0 <= ny < gray_image.shape[1]:
                        pixels_to_examine.append((nx, ny))
        
        return binary_mask

    
    def save_image(self):
        io.imsave(self.out_file, img_as_ubyte(self.image))


def run_region_grow(input_folder, output_folder):

    image_files = glob2.glob(os.path.join(input_folder, '*.png'))
    for image_file in image_files:
        print(f'writing {image_file.replace(input_folder, output_folder)}')
        RegionGrow(image_file, image_file.replace(input_folder, output_folder))



if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(f'Usage: python region_grow.py <input_folder> <output_folder>')
        print(f'Example: python region_grow.py /in/dir /out/dir')
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    run_region_grow(input_folder, output_folder)
    
