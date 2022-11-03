# Reverse Visual Search

![](google-earth-engine.png)

In many domains we are interested in finding artifacts that are similar to a query artifact. In this assignment you are going to implement a system that can find artifacts when the queries are visual. 

The type of queries you will test your system on are earth observatory images.  


## Create Dataset (20 points)

Earth Engine is a platform for petabyte-scale scientific analysis and visualization of geospatial datasets, both for public benefit and for business and government users. Earth Engine is free to use for research, education, and nonprofit use. To get started, please [sign up for Earth Engine access](https://developers.google.com/earth-engine/datasets).

Look at [this](https://developers.google.com/earth-engine/datasets/catalog/SKYSAT_GEN-A_PUBLIC_ORTHO_RGB) dataset and select an area of interest eg NYC. Using the Python API download to your Google drive few tiles (a tile is an square of thoudands of pixels eg 5000x5000) that covers the area of interest. 

You may want to consult this video to get you going

```{eval-rst}
.. youtube:: we79Gtt_o1w
```

## Create chips/patches (30 points)

In the following we use the term `chip`, an image of 256x256 pixels, to indicate the specific subset of the area of interest (also called a `patch`) that we reverse search on. The chip may be present together with hundreds of others in our dataset. Use the [pathify](https://pypi.org/project/patchify/) api to do that. You should now have hundreds of chips in your google drive. 

## Reverse Image Search - Feature extraction (30 points)

See [this example](https://aws.amazon.com/blogs/machine-learning/building-a-visual-search-application-with-amazon-sagemaker-and-amazon-es/) on how AWS is suggesting to build a reverse image search system.

Using ResNet-50 as the featuriser, extract for each chip a representative vector (2048 elements long).

## Reverse Image Search - Similarity Search  (20 points)

Using an chip as query, pick a [similarity search python library](https://github.com/currentslab/awesome-vector-search) to return the 10 most similar images to the query image. Select carefully the query image so that the results can be revealing of the ability of your representation and the search engine. 


Remember to document everything - setup instructions written in such a way so that someone else can replicate your assignment. You deliver this assignment as a Github URL and notebooks and/or markdown files should contain all the results. 
