# %% 0. Start
# %%% i. Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
import re # for parsing filenames
import sys
import math
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

# %%% ii. Import Internal Functions
from KRISP import run_model

# %%% iii. Directory Management
ipdmp_folder = os.getcwd()
try: # personal pc mode
    HOME = os.path.join("C:\\", "Users", "nicol", "Documents", "UoM", "YEAR 3", 
                        "Individual Project", "Downloads")
    os.chdir(HOME)
except: # uni mode
    HOME = os.path.join("C:\\", "Users", "c55626na", "OneDrive - "
                        "The University of Manchester", "Individual Project")
    os.chdir(HOME)

class_names = ["land", "reservoirs", "water bodies"]


