# Photomosaic

This Python project allows you to create a [photomosaic](https://en.wikipedia.org/wiki/Photographic_mosaic) from a target image and a large set of candidate images. 

+ **Distance/objective function**: I use a combination of the Euclidean distance and euclidean distance of the Sobel-transformed images. The edge detection is particularly important as it helps to match the edges of the tiles in the target image with the edges of the images in the set of images. There may be other methods worth investigating.

+ **Optimization algorithm**: creating a Photomosaic is a massive combinatorial optimization problem, so I currently use a cheap algorithm. First I take the target image and break it up into a grid of smaller pieces against which the candidate images are compared. For each candidate image from the database, I create a list of the best locations for that image according to the distance function. Then, I go through that list and place it in the grid if it is better than the current best. I repeat this process for all the images in the database. This clearly does not provide a global optimum, but it looks fine enough with enough images. An effective optimization algorithm might be important with smaller image databases.

+ **Test dataset**: I use the [Coco image dataset](https://cocodataset.org/) to test this code. It seems to work fairly well. I recommend that you use your own photos or images to put a personal touch on your creations, although admittedly it may require a massive number of photos.

## Examples


This is a photomosaic composed of images from the [Coco image dataset](https://cocodataset.org/).

![Obama composed of coco images](https://storage.googleapis.com/public_data_09324832787/obama_coco.png)

This was the original image:

![Obama photo original](https://storage.googleapis.com/public_data_09324832787/obama.png)


