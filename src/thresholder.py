
import numpy as np
from skimage import exposure, io, color
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte
import glob2
import sys
import os

class Threshold:
    """ Applies simple thresholding to a folder of PNG images. 
        Only keeps the largest object and removes all others. """

    def __init__(self, in_file, out_file, threshold):
        self.in_file = in_file
        self.out_file = out_file
        
        self.image = self.load_image()
        self.image = self.threshold(threshold)
        self.save_image()

    def load_image(self):
        im = io.imread(self.in_file)
        return im
        
    def threshold(self, thresh):
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

        return largest_component_mask
    
    def save_image(self):
        io.imsave(self.out_file, img_as_ubyte(self.image))


def run_thresholding(input_folder, output_folder, threshold):

    image_files = glob2.glob(os.path.join(input_folder, '*.png'))
    for image_file in image_files:
        print(f'writing {image_file.replace(input_folder, output_folder)}')
        p = Threshold(image_file, image_file.replace(input_folder, output_folder), threshold)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(f'Usage: python threshold.py <input_folder> <output_folder> <threshold>')
        print(f'Example: python threshold.py /in/dir /out/dir 0.123')
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    threshold = float(sys.argv[3])

    run_thresholding(input_folder, output_folder, threshold)





