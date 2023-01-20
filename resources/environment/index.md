---
title: Your Programming Environment
---

# Your Programming Environment

## Computational Environments

### Google Colaboratory (Option 1)

This course will be using Jupyter notebooks and we will be using the **free** CPU/GPU resources provided by [Google Colab](https://colab.research.google.com). The good news is that you have an account in Google Colab as most of you have your university gmail account. Please note if you login to colab with your university account you get to use the almost unlimited Gdrive storage facility. You will need Google Colab for all your projects so that you can demonstrate that your results can be replicated.  In addition Colab has many [features](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) that come handy. 

### Kaggle (Option 2)

You can use Kaggle as an alternative to Colab for all your projects. You guessed it right - all the projects in this course are in fact Kaggle competitions.  You can login with your gmail account. You can install by following the directions [here](https://github.com/Kaggle/kaggle-api) the Kaggle command line interface (CLI).  Not only you will get to compete (your ranking relative to others does not matter per se), but as you improve your knowledge over time you can revisit these competitions and see how your score improves.  There is one catch though - Colab offers more GPU time and it comes with multiple tiers of service for a few dollars per month. This can be handy for the heavier projects.  

### On your own machine or AWS EC2 (Option 3)

Please note that the [AWS Deep Learning AMIs](https://docs.aws.amazon.com/dlami/latest/devguide/ubuntu18-04.html) may not require installation or they may require update to the required version of TF/Pytorch.  Choose this option if you have money to spend but most if not all projects in this class will be able to be run in Google colab.

You need to create an isolated Python environment dedicated to this course. This is recommended as it makes it possible to have a different environment for each project, with potentially different libraries and library versions

#### Setup your virtual environment with standard Python

Follow the instructions [here](https://python-poetry.org/docs/) to install Poetry and [follow up with the basic usage instructions](https://python-poetry.org/docs/basic-usage/). Managing dependencies is one of the key challenges in data science. Poetry is a tool that helps you manage your dependencies.

Please ensure that you have used the virtualenvs.in-project configuration variable to create virtual environments **within** your project directory and that you know how to use .gitignore to avoid committing your virtual environment to your git repository.



## Git / Github

Git is the defacto standard when it comes to code version control. Learning basic git commands takes less than half an hour. However, to install git and understand the principle behind git, please go over Chapters 1 and 2 of the [ProGit book](https://git-scm.com/book/en/v2).

As we have discussed in the class you need to be able to publish your work in Github so you need to create a Github account. Then you will use the git client for your operating system to interact with github and iterate on your projects.  You may be using Kaggle or Colab hosted notebooks but the underlying technology that powers such web-frontends when it comes to committing the code and seeing version numbers in your screen is git.

In addition, almost no data science project starts in vacuum - there is almost always software that will be checked out of Github that you will need to modify. 

### How to work with a github repository in Colab

1. Fork the desired repository if this is not yours. For example go to https://github.com/ageron/handson-ml2 and press the Fork button. 
2. After forking you should see the repository appearing in your account. 
3. Click the green button `Clone or dowload`, click Use HTTPS and copy the field with the location of the repo your forked. 
4. Go to https://colab.research.google.com/ and login with your NJIT gmail account
5. In the window that pops up select Github. Accept the requested additional permission request for your NJIT gmail account. After Github and Colab connects you will be able to see the forked repo from your drop down menu of Repository. You will also see all the notebooks that start with a number e.g 01_the_machine_learning_landscape.ipynb. The number indicates the chapter number. 
6. Select to open the 01-*.ipynb notebook by clicking on it. You should see the notebook in your own colab account. Any change will be persisted in your github. 
7. Run the first cell. If you havent used Notebooks before, people with little programming experience will fall in love with them especially at this stage where you dont need to type new code. For a tutorial on how to use the notebooks in colab or in general open and run the notebook [Welcome to Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb).

## External Tools and Databases

### Elastic Search Environment Setup

For projects work you may need to install ES. Please note you are responsible for setting up the environment. For example to set up ES in Win10 you may follow [this](https://www.youtube.com/watch?v=hzaG2Uq60uw) guide but bear in mind that we cannot support any IT issues you may encounter in your laptop. You may decide to set up a development environment in AWS cloud 9 that is linux based for a small fee or taking advantage the free tier for _new_ AWS accounts (which is not free if you need EC2 instances outside of what the free tier provides).
