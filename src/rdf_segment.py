
import numpy as np
import joblib
from skimage import io
from skimage.util import view_as_windows
from skimage.io import imsave
# import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, measure
from skimage import io
import sys 
import os
import glob 


def extract_features(window):
    """Extract features from a 5x5x3 RGB window."""
    assert window.shape == (5, 5, 3), "Window should be a 5x5x3 RGB neighborhood"
    
    # Flatten the 5x5x3 window to extract features
    center_pixel_rgb = window[2, 2]
    center_r, center_g, center_b = center_pixel_rgb

    # Compute the mean and standard deviation across the window
    mean_rgb = np.mean(window, axis=(0, 1))  
    stddev_rgb = np.std(window, axis=(0, 1)) 

    # Return the features as a 1x9 vector
    return [center_r, center_g, center_b,
            mean_rgb[0], mean_rgb[1], mean_rgb[2],
            stddev_rgb[0], stddev_rgb[1], stddev_rgb[2]]

def generate_mask(image, model, window_size=3):
    """Generate a binary mask for the object using the trained model."""

    # Create sliding windows over the image (5x5 windows)
    windows = view_as_windows(image, window_shape=(window_size, window_size, 3), step=1)
    windows = windows.squeeze(axis=2)

    # Get the height and width of the image
    height, width, _ = image.shape
    
    # Extract features for all windows in one step
    # Flatten the windows into a 2D array of shape (num_windows, 5*5*3)
    num_windows = windows.shape[0] * windows.shape[1]
    feature_list = np.zeros((num_windows, 9))
    
    # Loop through all windows and compute the features
    for i in range(num_windows):
        # Get the window from the 2D grid of windows
        window = windows[i // windows.shape[1], i % windows.shape[1]]
        feature_list[i] = extract_features(window)
    
    # Predictions for all windows at once
    predictions = model.predict(feature_list)
    
    # Rebuild the mask from the predictions
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Reshape the predictions back into the original image grid
    mask[2:-2, 2:-2] = predictions.reshape(height - 4, width - 4)
    
    return mask


def post_process(image, kernel_size):
    """Close small holes, keep the largest component and remove all else."""
 
    # Join small holes using a NxN kernel
    selem = morphology.square(kernel_size)
    closed_image = morphology.closing(image, selem)
    
    # Label connected components and find the largest one
    labeled_image = measure.label(closed_image)
    region_props = measure.regionprops(labeled_image)
    
    # Find the largest region by area
    largest_region = max(region_props, key=lambda x: x.area)
    largest_region_label = largest_region.label
    
    # Create an empty image with the largest region
    final_image = labeled_image == largest_region_label
    
    return final_image.astype(np.uint8)


def segment(input_folder, output_folder, model_folder):

    # Load the trained model
    clf = joblib.load(os.path.join(model_folder, 'random_forest_model.pkl'))

    image_files = glob.glob(os.path.join(input_folder, '*.png'))

    for image_file in image_files:    
        image = io.imread(image_file)
        mask = generate_mask(image, clf, window_size=5)
        processed_image = post_process(mask, 3)
        filename = image_file.replace(input_folder, output_folder)
        print(f'writing {filename}')
        # Convert mask to 0-255 range for saving
        imsave(filename, processed_image * 255) 


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(f'Usage: rdf_segment.py <input_folder> <output_folder> <model_folder>')
        print(f'Example: python rdf_segment.py /in/dir /out/dir /model/dir')
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    model_folder = sys.argv[3]

    segment(input_folder, output_folder, model_folder)

