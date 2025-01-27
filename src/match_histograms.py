
from skimage import exposure
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
import glob2
import os
import sys


def histogram_matching(source_img, reference_img):
    """Perform histogram matching of the source image to match the reference image."""
    # Normalize the images to float (0-1 range)
    source_img = img_as_float(source_img)
    reference_img = img_as_float(reference_img)
    matched_img = exposure.match_histograms(source_img, reference_img, channel_axis=-1)
    return matched_img


def run_histogram_matching(input_folder, output_folder, reference_path):
    """Apply histogram matching to all images in a directory.
       One of the images in the directory should be identified as the reference image."""
    input_images = glob2.glob(os.path.join(input_folder, '*.png'))
    input_images.remove(reference_path)
    reference_image = imread(reference_path)
    for input_image in input_images:
        source_image = imread(input_image)
        matched_img = histogram_matching(source_image, reference_image)
        print(f'writing: ', input_image.replace(input_folder, output_folder))
        imsave(input_image.replace(input_folder, output_folder), img_as_ubyte(matched_img))
    imsave(reference_image.replace(input_folder, output_folder), img_as_ubyte(reference_image))


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(f'Usage: python match_historgrams.py <input_folder> <output_folder> <reference_image>')
        print(f'Example: python match_historgrams.py /in/dir /out/dir /in/dir/filename.pdf')
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    reference_path = sys.argv[3]

    run_histogram_matching(input_folder, output_folder, reference_path)


