import os
import sys
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')

class ImageViewer:
    """Simple image viewer for visually flipping through a set of Landsat images
       in a directory. Left and right keys flip through images in the folder."""

    def __init__(self, image_folder, max_size=(400, 400)):
        self.image_folder = image_folder
        # max_size to downsample images
        self.max_size = max_size
        self.image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')) +
                                  glob.glob(os.path.join(image_folder, '*.tif')))
        
        # downsample images for efficiency
        self.images = [self.load_and_downsample(image_path) for image_path in self.image_files]
        
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        
        self.current_image = 0
        self.im = self.ax.imshow(self.images[self.current_image])
        self.fig.suptitle(f'{os.path.basename(self.image_files[self.current_image])}')
        
        # connect keyboard event for flip through
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # connect close event
        self.fig.canvas.mpl_connect('close_event', self.on_close)


    def load_and_downsample(self, image_path):
        img = Image.open(image_path)
        # downsample image if it is larger than the max size
        img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        return img


    def display_image(self, index):
        """Display the image at the given index."""
        image = self.images[index]
        # update image without clearing axes
        self.im.set_data(image)
        self.fig.suptitle(f'{os.path.basename(self.image_files[index])}')
        # use draw_idle() for fast rendering
        self.fig.canvas.draw_idle()


    def on_key_press(self, event):
        """Key handler for left and right keys."""
        if event.key == 'right':
            self.current_image = (self.current_image + 1) % len(self.images)
            self.display_image(self.current_image)
        elif event.key == 'left':
            self.current_image = (self.current_image - 1) % len(self.images)
            self.display_image(self.current_image)


    def on_close(self, event):
        """Cleanup function on window close."""
        print("Closing the viewer...")
        plt.close('all')


    def run(self):
        """Display the window and start the event loop."""
        plt.show()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f'Usage: python image_viewer.py <input_folder>')
        print(f'Example: python image_viewer.py /in/dir')
        sys.exit(1)

    folder_path = sys.argv[1]
    viewer = ImageViewer(folder_path)
    viewer.run()
