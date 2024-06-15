Usage
=====

Installation
------------

It is possible to clone the repository containing the code here:

.. code-block:: console

    git clone git@github.com:chiaratomaiuolo/xrayreco.git

The NNs have been trained using solid state detector simulations, the input files
were `HDF5` files containing the detected signal, the MC truth and a set of 
information about the simulation parameters. This project contains the necessary
functions for the preprocessing of this kind of data, it uses function from `hexsample`.