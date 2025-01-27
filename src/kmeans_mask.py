
from sklearn.cluster import KMeans
from skimage.util import img_as_ubyte
from skimage import exposure 
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage import io, color
import numpy as np
import glob2
import sys
import os

class Clustering:
    """Segments images in a directory using K-Means clustering.  
       The number of clusters is expected from the user.  The largest object
       is retained after clustering."""

    def __init__(self, in_file, out_file, n_clusters=3):
        self.in_file = in_file
        self.out_file = out_file
        self.n_clusters = n_clusters
        self.image = self.load_image()
        self.labels = None
        self.mask = None
        self.seed = self.seed(0.1)
        self.colors = self._create_label_colors()
        self.image = self.cluster()
        self._create_mask()
        self.save_image()

    def _create_label_colors(self):
        np.random.seed(42)
        return np.random.randint(0, 256, size=(self.n_clusters, 3))

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
        # Retain the largest object
        segmented_largest_component = np.zeros_like(gray_image)
        segmented_largest_component[largest_component_mask] = gray_image[largest_component_mask]
        # Compute the distance transform
        dist_transform = distance_transform_edt(segmented_largest_component)
        # Find the seed point (pixel with maximum distance)
        seed_point = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        y, x = seed_point

        return (y,x)


    def cluster(self):
        # Reshape the image into a 2D array of pixels
        pixels = self.image.reshape(-1, 3)
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(pixels)
        # Get the labels and reshape back to the original image shape
        self.labels = kmeans.labels_.reshape(self.image.shape[0], self.image.shape[1])


    def _create_mask(self):
        # Find the object color
        object_color = self.labels[self.seed[0], self.seed[1]]
        # Create a mask from the object colored regions
        self.mask = self.labels == object_color
        # Label the connected components in the binary mask
        labeled_mask = label(self.mask)
        # Get properties of the labeled regions
        regions = regionprops(labeled_mask)
        # Find the largest region by area
        largest_region = max(regions, key=lambda region: region.area)
        # Update the mask to retain only the largest component
        self.mask = labeled_mask == largest_region.label


    def save_image(self):
        io.imsave(self.out_file, img_as_ubyte(self.mask.astype(np.uint8)*255))


def run_clustering(input_folder, output_folder, number_of_clusters):

    image_files = glob2.glob(os.path.join(input_folder, '*.png'))
    for image_file in image_files:
        print(f'writing {image_file.replace(input_folder, output_folder)}')
        Clustering(image_file, image_file.replace(input_folder, output_folder), number_of_clusters)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(f'Usage: python kmeans_mask.py <input_folder> <output_folder> <number_of_clusters>')
        print(f'Example: python kmeans_mask.py /in/dir /out/dir 3')
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    number_of_clusters = int(sys.argv[3])

    run_clustering(input_folder, output_folder, number_of_clusters)






