# %% load a file

import h5py
import numpy as np  

dataLoc = "/Users/benjaminscholl/Dropbox/projects/2poptostim/mouseSCexamples/"
fname = "TSeries-05062026-1341-002.h5"
dat = h5py.File(dataLoc + fname, "r")


# %%
