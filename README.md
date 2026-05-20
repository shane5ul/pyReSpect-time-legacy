# pyReSpect-time

Extract continuous and discrete relaxation spectra from stress relaxation modulus G(t). The papers which describe the method and test cases are:

+ Shanbhag, S., "pyReSpect: A Computer Program to Extract Discrete and Continuous Spectra from Stress Relaxation Experiments", Macromolecular Theory and Simulations, **2019**, 1900005 [doi: 10.1002/mats.201900005].
+ Takeh, A. and Shanbhag, S. "A computer program to extract the continuous and discrete relaxation spectra from dynamic viscoelastic measurements", Appl. Rheol. **2013**, 23, 24628. 

## Files

### Code Files

This repository contains two python modules `contSpec.py` `discSpec.py`. They extract the continuous and discrete relaxation spectra from a stress relaxation data. (t versus G(t) experiment or simulation).

It containts a third module `common.py` which contains utilities required by both `contSpec.py` and `discSpec.py`.

In addition to the python modules, a jupyter notebook `interactContSpec.ipynb` is also provisionally included. This allows the user to experiment with parameter settings interactively.

### Input Files

The user is expected to supply two files:

+ `inp.dat` is used to control parameters and settings
+ `Gt.dat` which contains two columns of data `t` and `G(t)` [July 2023: optional third column specifying weight of datapoints]

### Output Files

Text files containting output from the code are stored in a directory `output/`. These include a fit of the data, the spectra, and other files relevant to the continuous or discrete spectra calculation. 

Graphical and onscreen output can be suppressed by appropriate flags in `inp.dat`.

### Test Files

A bunch of test files are supplied in the folder `tests/`. These data are described in the paper:
Shanbhag, S., "pyReSpect: A Computer Program to Extract Discrete and Continuous Spectra from Stress Relaxation Experiments" which will appear in Macromolecular Theory and Simulations in **2019**.

## Usage

Once `inp.dat` and `Gt.dat` are furnished, running the code is simple.

To get the continuous spectrum:

`python3 contSpec.py`

The **continuous spectrum must be extracted before the discrete spectrum** is computed. The discrete spectrum can then be calculated by

`python3 discSpec.py`

### Interactive Mode

The interactive mode offers a "GUI" for exploring parameter settings. To launch use `jupyter notebook interactContSpec.ipynb`.

### Pre-requisites

The numbers in parenthesis show the version this has been tested on.

python3 (3.12)
numpy (2.4)
scipy (1.17)

## History

The code is based on the Matlab program [ReSpect](https://www.mathworks.com/matlabcentral/fileexchange/40458-respect), which extract the continuous and discrete relaxation spectra from frequency data, G*(w). Work was supported by National Science Foundation DMR grants number 0953002 and 1727870.

### Major Upgrade: March-April 2019
+ added ability to infer plateau modulus G0; modified all python routines and reorganized inp.dat
+ use a Bayesian formulation to infer uncertainty in the continuous spectrum
+ currently keeping old method to determine critical lambda, but using a far more efficient method (3-4x savings in compute time)
+ made discSpec.py compliant with G0

### Major Upgrade: December 2018

+ moved all common imports and definitions to common; made completely visible
+ in discSpec(): added a NLLS routine to optimize tau; use previous optima as initial guesses for final tau; this greatly improved the quality of fits.


### Major Upgrade: August 2018

#### Continuous Spectrum
+ original program with n = 100 and lambda = 20 pts with clean 1 mode data took ~33s.
+ prestore kernMat: evaluation of kernel (meshgrid S, T, and hs) by prestoring kernMat ~12.5s (~3x speed gain)
+ improved least_squares setting by incorporating jacobianLM; ~6s (2x gain)
+ lcurve coarser (auto) mesh, robust criterion, and interpolation ~3.5s (~1.75x gain)
+ Total gain in speed as a consequence of these improvements is 33s -> 3.5s a nearly 10x gain!
+ making jupyter interact compliant

#### Discrete Spectrum
+ AutoMagic Mode: need only par verbose and plotting flags; auto Nopt
+ switching to nnls as default fitting engine
+ changed older MaxwellModes and LLS -> nnls
+ some printing modifications
+ hardcoding prune = True everywhere; doesn't seem to be use case otherwise
+ making jupyter interact compliant
