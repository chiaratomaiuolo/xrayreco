.. xray_hit_reconstruction documentation master file, created by
   sphinx-quickstart on Wed Jun  5 14:40:43 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xrayreco
=========================
`xrayreco` is a project that aims to reconstruct incident X-rays energies and
hit position on a solid state detector using a Neural Networks. 

Overview
--------
In the following package the training and evaluation of Neural Networks for
reconstructing the energy and the position of X-ray signals on a solid-state
detector having an hexagonal grid of pixels as reaodut.  
The training weights have been saved and stored in the repository, so that the 
NNs can be used for future predictions of this kind of signals, the results are
stored in `HDF5` files.


Contents
--------
.. toctree::
   usage
   dataprocessing
   nnmodels
   fitfacilities
   predfile



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
