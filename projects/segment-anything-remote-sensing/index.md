# Segment Anything Model for Remote Sensing Applications

Lately, we have observed the advent of very large foundational models in NLP applications - sometimes these models result in  hundreds of billions of parameters - for example the [GPT-3](https://arxiv.org/abs/2005.14165) model that has 175 billion parameters. 

In this project you will be asked to explore the analogous solution for computer vision. We will use the  Segmenting Anything Model (SAM) to segment sidewalks and other features from satellite imagery (remote sensing).

## Background

![](images/sam.png)

Read and experiment with [the SAM implementation](https://segment-anything.com/) and [the SAM paper](https://arxiv.org/abs/2101.04703).


## Development environment setup 

Use the [example docker container](https://github.com/pantelis/artificial-intelligence) if you have access to an NVIDIA GPU or use Colab to set up the development environment that uses pip to install dependencies.

```{note}
Avoid working with conda environments as they are not compatible with optimized NVIDIA containers and they also have issues with Colab. Always prefer pip-based tooling. For docker containers you dont need to create virtual environments as the container is isolated from the host but if you somehow want you can use [pipenv](https://pipenv.pypa.io/en/latest/) or simply the standard library `venv`. 

```
On top of plain python scripting you will also need python notebooks to experiment and visualize the SAM model outputs as well as geospatial data. The suggested container ships with [Jupyter](https://jupyter.org/) and if you use [Google Colab](https://colab.research.google.com/) you are obviously using notebooks.  

A desktop version of a GIS tool such as [QGIS](https://qgis.org/en/site/) is also recommended if you like to work with remote sensing applications.


## Replicate SAM implementation for satellite imagery

The SAM implementation for remote sensing applications is based on [this package](https://samgeo.gishub.org/). Go over the following video and replicate the workflow in your environment for the imagery used by the author.    

```{eval-rst}
.. youtube:: YHA_-QMB8_U
   :width: 100%
   :height: 400
   :align: center
   
```

The notebook based environment makes it very easy to experiment with the SAM model and the workflow. 

This [desktop QGIS plugin](https://github.com/BjornNyberg/Geometric-Attributes-Toolbox/wiki/User-Guide#segment-anything-model) is also available to help you experiment with the SAM model.


## Finetune the SAM model for the sidewalks dataset

Use the sidewalk dataset (this is currently being prepared and will be released soon) and finetune the SAM model for this dataset. Follow these instructions to do so. 

```{eval-rst}
.. youtube:: 83tnWs_YBRQ
   :width: 100%
   :height: 400
   :align: center
```

Consider loss functions and segementation quality metrics that are suitable for roads and sidewalks. You can see [this reference for guidance](https://www.sciencedirect.com/science/article/pii/S1569843222003478).  Avoid going into a rabbit hole of developing your own metrics - there are many metrics that are suitable for this task and you just need to be aware of them and use them. These include: 

- [Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index)
- [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- [F1 score](https://en.wikipedia.org/wiki/F-score)

One interesting research opportunity is to consider the fact that there is no such thing as sidewalk that is not next to a road / street. Can we improve the model by considering contrastive loss functions that take into account the fact that sidewalks are always next to roads when sidewalks are present and roads can also be present without sidewalks? 






