#!/usr/bin/env python
# -*- coding: utf-8 -*-
##################################### Help ####################################
"""Simple KNN implementation
Usage:
  knn.py [options]
  knn.py --help
Options:
  -a                          Additional option
  -h --help                   Show this screen.
"""


################################### Imports ###################################
from docopt import docopt
import numpy as np
import random

################################## Functions ##################################

################################### Classes ###################################

##################################### Main ####################################
# Import user input
arguments = docopt(__doc__)
DATA_FILE_NAME = "/home/koala/Documents/Scripts/KNN/KNN/glass.data"

# Load data
data_file = open(DATA_FILE_NAME)
data = data_file.readlines()
data = np.array([line[:-1].split(",") for line in data], dtype=np.float64)

print(data)