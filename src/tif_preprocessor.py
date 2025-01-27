
import os
import sys
import numpy as np
from PIL import Image
from skimage import io
import glob2

class TIFPreprocessor:
    """ Reads TIF images from input directory, 
        extracts the image date from the filename,
        pads smaller images with black to the largest image size,
        (optionally) crops to region of interest, 
        writes as PNG images to output directory. """

    def __init__(self, in_dir, out_dir, crop=None):

        self.in_dir = in_dir
        self.out_dir = out_dir

        self.image_files = sorted(glob2.glob(os.path.join(in_dir, '*.png')) + 
                                  glob2.glob(os.path.join(in_dir, '*.tif')))

        # extract the dates from file names
        self.image_dates = [x.split('_')[-4] for x in self.image_files]

        # find largest image dimensions
        self.max_width, self.max_height = self._get_max_image_size()

        self.images = self.load_images()
        if crop:
            self.crop_images(crop)
        
        self.save_images()


    def _get_max_image_size(self):
        max_width = max_height = 0
        print('reading images ', end='')
        for image_file in self.image_files:
            with Image.open(image_file) as img:
                print('.', end='')
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
        print()
        return max_width, max_height


    def _pad_image(self, img, target_width, target_height):
        # pad smaller images with black to match maximum width and height
        width, height = img.size
        new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        
        # calculate position to paste the original image at the center
        left = (target_width - width) // 2
        top = (target_height - height) // 2
        
        new_img.paste(img, (left, top))
        return np.array(new_img)
    

    def crop_images(self, crop_box):
        # crop to the region of interest
        left, top, right, bottom = crop_box
        self.images = self.images[top:bottom, left:right, :, :]


    def load_images(self):
        # load all images and return as a 4D numpy array (height, width, channels, image_num)
        image_list = []
        
        for image_file in self.image_files:
            with Image.open(image_file) as img:
                # convert image to RGB
                img = img.convert('RGB')
                # pad image to the target size
                padded_img = self._pad_image(img, self.max_width, self.max_height)                
                # append image to the list
                image_list.append(padded_img)
        
        # stack and return all images 
        return np.stack(image_list, axis=-1)
    
    
    def save_images(self):
        num_images = self.images.shape[-1]
        for i in range(num_images):
            filename = os.path.join(self.out_dir, f'{self.image_dates[i]}.png')
            print(f'writing {filename}')
            io.imsave(filename , self.images[:,:,:,i])


if __name__ == "__main__":

    if not len(sys.argv) in [3, 7]:
        for a in sys.argv:
            print(a)
        print(f'Usage: python tif_preprocess.py <input_folder> <output_folder> <optional: crop_region left top right bottom>')
        print(f'Example: python tif_preprocessor.py /in/dir /out/dir 100 200 150 250')
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if len(sys.argv) == 7:
        left = int(sys.argv[3])
        top = int(sys.argv[4])
        right = int(sys.argv[5])
        bottom = int(sys.argv[6])
        crop_box = (left, top, right, bottom)
    else:
        crop_box = None

    pp = TIFPreprocessor(input_folder, output_folder, crop=crop_box)
    print(f'Image dimensions: {pp.images.shape}')

