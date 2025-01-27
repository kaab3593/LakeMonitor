
import os
import sys
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('MacOSX')



class PixelCollector:
    """Pixel collector is a tool to extract local features from a set of images in a 
       directory.  It captures the user click, computes the local features, and saves the 
       results in a CSV file.  The MODE selects for FOREGROUND (f) or BACKGROUND (b).  
       If mode is not set, user clicks on an image are not recorded in the CSV.  
       Left and rigth keys are used to flip through images in the directory."""

    def __init__(self, image_folder, output_folder):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')) +
                                  glob.glob(os.path.join(image_folder, '*.tif')))
        self.images = [Image.open(image_path) for image_path in self.image_files]

        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.current_image = 0
        self.im = self.ax.imshow(self.images[self.current_image])
        self.fig.suptitle(f'{os.path.basename(self.image_files[self.current_image])}')

        # initialize mode
        self.mode_text = self.ax.text(0.5, -0.1, 'Mode: None', ha='center', va='top', transform=self.ax.transAxes, fontsize=12, color='black')
        self.foreground_mode = None

        # store clicked pixel locations
        self.clicked_pixels = []

        # CSV file to store the features
        self.csv_filename = os.path.join(self.output_folder, 'features.csv')
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Image Filename", "Center RGB R", "Center RGB G", "Center RGB B",
                             "Mean RGB R", "Mean RGB G", "Mean RGB B", 
                             "StdDev RGB R", "StdDev RGB G", "StdDev RGB B", "Mode"])

        # connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)  # for keyboard navigation
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)  # for mouse click

    def update_mode_text(self):
        """Update the mode text label on the figure."""
        if self.foreground_mode == 'foreground':
            mode_label = 'Mode: Foreground'
        elif self.foreground_mode == 'background':
            mode_label = 'Mode: Background'
        else:
            mode_label = 'Mode: None'  # No mode selected
        # Update the mode text on the figure
        self.mode_text.set_text(mode_label)
        self.fig.canvas.draw_idle()  # Redraw the canvas to reflect the updated text

    def on_key_press(self, event):
        """Handle key press events for navigation and mode toggling."""
        if event.key == 'right':
            self.current_image = (self.current_image + 1) % len(self.images)
            self.display_image(self.current_image)
        elif event.key == 'left':
            self.current_image = (self.current_image - 1) % len(self.images)
            self.display_image(self.current_image)
        elif event.key == 'f': 
            self.foreground_mode = 'foreground'
            self.update_mode_text() 
        elif event.key == 'b': 
            self.foreground_mode = 'background'
            self.update_mode_text()  
        else:
            # If any other key is pressed, toggle the mode off
            if event.key not in ['f', 'b']:
                self.foreground_mode = None
                self.update_mode_text() 


    def on_click(self, event):
        """Handle mouse clicks to capture pixel locations and extract features."""
        # Ignore clicks outside the image area or when the mode is not set
        if event.inaxes != self.ax or self.foreground_mode is None:
            return

        # Get the pixel coordinates relative to the image
        # Convert to integer pixel location
        x_pixel = int(event.xdata)  
        y_pixel = int(event.ydata)

        # Get the image corresponding to the current view
        image = np.array(self.images[self.current_image])
        # Extract the 5x5 neighborhood around the clicked pixel
        neighborhood = self.get_neighborhood(image, x_pixel, y_pixel, window_size=5)
        # Extract features from the neighborhood using the extract_features method
        features = self.extract_features(neighborhood)
        # Add the mode indicator (1 for foreground, 0 for background) to the feature vector
        mode_indicator = 1 if self.foreground_mode == 'foreground' else 0
        features.append(mode_indicator)
        # Save the features to the CSV file
        self.save_features_to_csv(features)


    def extract_features(self, window):
        """Extract features from a 5x5x3 RGB window."""
        # Ensure the window is of shape (5, 5, 3)
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


    def get_neighborhood(self, image, x, y, window_size=5):
        """Extract the RGB values of the neighborhood surrounding the center pixel."""
        # Define the half window size
        half_window = window_size // 2

        # Extract the region surrounding the center pixel
        min_x = x - half_window
        max_x = x + half_window + 1
        min_y = y - half_window
        max_y = y + half_window + 1

        # Extract the neighborhood (a 5x5 block centered at (x, y))
        neighborhood = image[min_y:max_y, min_x:max_x]

        return neighborhood


    def save_features_to_csv(self, features):
        """Save the extracted features to a CSV file."""
        # features is now a list of 10 values (9 features + 1 mode indicator)
        center_r, center_g, center_b, mean_r, mean_g, mean_b, stddev_r, stddev_g, stddev_b, mode_indicator = features

        # Save the features to the CSV file
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(self.image_files[self.current_image]),
                            center_r, center_g, center_b,
                            mean_r, mean_g, mean_b,
                            stddev_r, stddev_g, stddev_b,
                            mode_indicator])


    def display_image(self, index):
        """Display the image at the given index."""
        image = self.images[index]
        # Update image data without clearing the axes
        self.im.set_data(image)
        self.fig.suptitle(f'{os.path.basename(self.image_files[index])}')
        # Use draw_idle() for faster rendering
        self.fig.canvas.draw_idle()  

    def run(self):
        """Display the window and start the event loop."""
        plt.show()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(f'Usage: python pixel_collector.py <input_folder> <output_folder>')
        sys.exit(1)

    folder_path = sys.argv[1]
    output_folder = sys.argv[2]
    viewer = PixelCollector(folder_path, output_folder)
    viewer.run()

