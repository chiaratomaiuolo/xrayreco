Usage
=====

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

