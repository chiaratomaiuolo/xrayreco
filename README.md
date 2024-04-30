# X-ray energy and hit position reconstruction on an hexagonal grid
## Goal
The following repository contains the dataset and the code used for training a FFNN for the reconstruction of the _energy_ and _hit position in cartesian coordinates (x,y)_ of X-rays emitted by a Cu source in the K-line hitting an hexagonal pixel grid. 

## Datasets
Training and test datsets are in `.h5` format and have been simulated on a grid, varying two parameters:
- the layout of the hexagonal grid: ODD_R, ODD_Q, EVEN_R, EVEN_Q (see: https://www.redblobgames.com/grids/hexagons/);
- the electronic noise of every pixel in ENC, the investigated values are in the tuple: [0, 20, 40].

## Simulation parameters
--Detector scheme--
The fundamental pieces of the simulations with their parameters are:
- The _source beam_: a Gaussian beam centered in (0,0) with $\sigma = 0.1 \text{ cm}$ _not sure, maybe it is too much, there is another dataset with 0.02 cm_. The width of the Gaussian is much larger than the pitch (that is 50 $\mu \text{m}$) in order to be sure to irradiate circa uniformely the central pixel (here 2D hist of hits with the hexagonal contour).
- The _Silicon active medium_: with its _thickness_, that is set to a default value of 3 mm (the others are constant physical parameters).
- The _readout hexagonal grid_: characterized by the _pitch_ between hexagon centers, by the aforementioned _layout_ and _noise_.

## Data preprocessing
Before the NN training, data follows a procedure of _conversion and preprocessing_: the `.h5` is taken and the target quantities, `pha, col, row`, are extrapolated, rearranged in a standard ordering and their coordinates rescaled. After this first stage of preprocessing, for every event, a row in a `.txt` file is filled.



saved in a `.txt` file, where every row contains the information for a single photon event;


