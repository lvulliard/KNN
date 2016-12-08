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
from collections import defaultdict
import numpy as np
import sys, pickle, random

################################## Functions ##################################

################################### Classes ###################################

##################################### Main ####################################
# Import user input
arguments = docopt(__doc__)
print("Hello world.")