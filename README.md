SFselect
========

Learning Natural Selection from the Site Frequency Spectrum


This repoository is forked from https://github.com/rronen/SFselect.  It includes two main programs:

1) SFselect.py -- a standalone program for applying pre-trained SVMs of the site frequency spectrum (SFS) to allele frequency data. The output is for each sliding genomic window a probability under the model that the window is evolving under a sweep.

For more details on using SFselect.py, see http://bioinf.ucsd.edu/~rronen/sfselect.html

2) SFselect\_train.py -- a program for training SVMs of the site frequency spectrum (SFS) to classify regions evolving neutrally from those evolvign under a hard selective sweep. This program requires as input simulated population data (can be generated by simulators like ms, msms, etc). See 'params.py' for setting the simulation parameters (from which the data file names are constructed, among other things).


###Dependencies 
numpy, matplotlib, scikits-learn (tested with v0.13)
