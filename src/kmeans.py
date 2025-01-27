
from sklearn.cluster import KMeans
from skimage import io
from skimage.util import img_as_ubyte
import numpy as np
import glob2
import sys
import os

class Clustering:
    """Segments images in a directory using K-Means clustering.  
       The number of clusters is expected from the user.  This script writes
       the results of the clustering with label colors but does not generate a 
       binary mask.  To generate a binary mask, use kmeans_mask.py."""

    def __init__(self, in_file, out_file, n_clusters=3):
        self.in_file = in_file
        self.out_file = out_file
        self.n_clusters = n_clusters
        self.image = self.load_image()
        self.colors = self._create_label_colors()
        self.image = self.cluster()
        self.save_image()

    def _create_label_colors(self):
        np.random.seed(42)
        return np.random.randint(0, 256, size=(self.n_clusters, 3))

    def load_image(self):
        im = io.imread(self.in_file)
        return im
        
    def cluster(self):
        # Reshape the image into a 2D array of pixels
        pixels = self.image.reshape(-1, 3)
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(pixels)
        # Get the labels and reshape back to the original image shape
        labels = kmeans.labels_.reshape(self.image.shape[0], self.image.shape[1])
        # Create an RGB image from label image
        segmented_image = np.zeros_like(self.image)
        # Assign random colors to each label
        for i in range(self.n_clusters):
            cluster_mask = (labels == i)
            segmented_image[cluster_mask] = self.colors[i]
        return segmented_image

    def save_image(self):
        io.imsave(self.out_file, img_as_ubyte(self.image))


def run_clustering(input_folder, output_folder, number_of_clusters):

    image_files = glob2.glob(os.path.join(input_folder, '*.png'))
    for image_file in image_files:
        print(f'writing {image_file.replace(input_folder, output_folder)}')
        Clustering(image_file, image_file.replace(input_folder, output_folder), number_of_clusters)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(f'Usage: python kmeans.py <input_folder> <output_folder> <number_of_clusters>')
        print(f'Example: python kmeans.py /in/dir /out/dir 4')
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    number_of_clusters = int(sys.argv[3])

    run_clustering(input_folder, output_folder, number_of_clusters)






