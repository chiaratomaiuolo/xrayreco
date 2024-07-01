Usage
=====

Overview
--------
In the following package the training and evaluation of Neural Networks for
reconstructing the energy and the position of X-ray signals on a solid-state
detector having an hexagonal grid of pixels as reaodut.  
The training weights have been saved and stored in the repository, so that the 
NNs can be used for future predictions of this kind of signals.


Installation
------------

It is possible to clone the repository here:

.. code-block:: console

    git clone git@github.com:chiaratomaiuolo/xrayreco.git

The NNs have been trained using solid state detector simulations, stored in the 
_dataset_ directory. The datasets are in `HDF5` format, the structure is read using
dedicated functions from the `hexsample` package (required).

Tutorial
--------
In the directory 

.. code-block:: console

    /xrayreco/xrayreco/tutorial

There is Jupyter notebook `nn_trial1.ipynb` where is is possible to see the training
process of a NN from the opening of a training dataset to the evaluation of the
predicted pdfs.

