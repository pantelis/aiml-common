<!-- To install virtualenv
```
python3 -m pip install --user -U virtualenv
```
In the root of your project install the virtual environment. 

```
cd my_project
python3 -m virtualenv my_env
source my_env/bin/activate # on Linux or macOS
.\my_env\Scripts\activate  # on Windows
```
You can now install the packages/modules you need:

```
python3 -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn
```

If you created a virtualenv, you need to register it to Jupyter and give it a name:

```
python3 -m ipykernel install --user --name=python3
```

You are now able to use a jupyter or jupyterlab interface in your own laptop. 

#### Setup your virtual environment with Anaconda Python

When using Anaconda:

    $ conda create -n p38 python=3.8 anaconda
    $ conda activate py38

This creates a fresh Python 3.8 environment called `py38` (you can change the name if you want to), and it activates it. This environment contains all the scientific libraries that come with Anaconda. This includes all the libraries we will need (NumPy, Matplotlib, Pandas, Jupyter and a few others), except for TensorFlow, so let's install it:

    $ conda install -n py38 -c conda-forge tensorflow
    $ conda install -n py38 -c conda-forge tensorflow-gpu (if you have a computer with an NVIDIA GPU which is a must have).

This installs the latest version of TensorFlow available for Anaconda (which is usually *not* the latest TensorFlow version) in the `py37` environment (fetching it from the `conda-forge` repository). 

```{note}
Next, you can optionally install Jupyter extensions. These are useful to have nice tables of contents in the notebooks, but they are not required.

```
pip install jupyter_contrib_nbextensions # with standard pip
conda install -n py38 -c conda-forge jupyter_contrib_nbextensions # with anaconda
```

If you want to use the Jupyter extensions (optional, they are mainly useful to have nice tables of contents), you first need to install them:

    $ jupyter contrib nbextension install --user

Then you can activate an extension, such as the Table of Contents (2) extension:

    $ jupyter nbextension enable toc2/main

Okay! You can now start Jupyter, simply type:

    $ jupyter notebook

This should open up your browser, and you should see Jupyter's tree view, with the contents of the current directory. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree). Click on `index.ipynb` to get started!

Note: you can also visit [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions) to activate and configure Jupyter extensions.

``` -->