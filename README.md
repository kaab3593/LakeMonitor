
# Lake Title

**Description:**  

We compare different segmentation methods to extract surface area and boundary extent from Landsat satellite images of water bodies such as lakes and reservoirs.  The following segmentation methods are implemented: simple thresholding, region growing, k-means clustering, and random decision forests.  We use histogram matching to reduce the lighting variations due to reflectances from the Sun's position in all methods.  Other preprocessing and postprocessing steps used in each method, as well as other implemented tools such as the image viewer and pixel collector, are described in the example Jupyter Notebook `examples.ipynb`.

Each method reads PNG images from an input folder and writes the output in the same form to an output folder.  Image names are expected to have the image date in YYYYMMDD format.  A charting script processes the output folders and generates the plots of area and boundary lengths. 

## Methods

- **Thresholding:** A user-provided threshold value between 0 and 1 is applied to the scene to obtain the binary mask.  To enhance the contrast of the scene we apply historgram equalization as a preprocessing step.  The largest connected component is retained after the thresholding.

- **Region growing:** Region growing method is essentially an extension of the thresholding method. After thresholding, we use distance transform to locate a seed position from which we grow the object region until a different color is encountered.

- **K-means clustering:** We use K-means clustering to group pixels in the scene based on their color so that similarly colored pixels are in the same group. The algorithm selects k random points in the image to represent the cluster centers. Each pixel in the image is assigned to the cluster, whose centroid is closest in color. After assigning all pixels, the centroids are updated to be the average color of the pixels in each cluster. The assignment and averaging steps are repeated until the centroid stop changing. When the clustering process is complete, the image is segmented into k regions, each with similar colors grouped together.

- **Random forests:** Random froests work by classifying each pixel into different regions based on features, such as color or texture of surrounding pixels. We train and evaluate random forest models using data we extract from different scenes with the pixel_collector tool.  The pixel collector tool labels each data point as *foreground* or *background*.  We use color based statistics in a 5x5 neighborhood, which achieves a 99.92% accuracy on the validation set.  The random forest algorithm creates multiple decision trees, each trained on a random subset of features.  Each tree looks at different parts of the image and makes its own decision on which group a pixel might belong.  On a new image, each pixel is classified into foreground or background groups by all the decision trees.  Based on the votes of individual trees, each pixel is assigned to the segment with the most votes.


## Usage

See the notebook `examples.ipynb`.

Scripts can be run independently from the command line.


## Visuals

Below is the resulting area and boundary length plots from the Random Forest segmentation.

![Random Forest Segmentation](plots/landsat_rdf.pdf)


## License

This project is licensed under the Academic Free License ("AFL") v. 3.0 - see the [LICENSE](license) file for details.
