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
class KNN:
	"""A K-nearest neighbors regression"""
	
	def __init__(self, pts):
		self.points = pts

	def predict(self, pt):
		dist = np.array([np.linalg.norm(data[refPoint,1:10] - data[pt,1:10]) for refPoint in self.points])
		dist_sort = dist.argsort()
		nearest_points = [self.points[i] for i in dist_sort[:K]]
		return sum(data[nearest_points,10])/K
		


##################################### Main ####################################
# Seed for reproductibility
random.seed(1)

# Import user input
arguments = docopt(__doc__)
DATA_FILE_NAME = "/home/koala/Documents/Scripts/KNN/KNN/glass.data"
NB_CROSSVAL = 10
K = 3

# Load data
data_file = open(DATA_FILE_NAME)
data = data_file.readlines()
data = np.array([line[:-1].split(",") for line in data], dtype=np.float64)
# Shuffle the data
neworder = range(len(data))
random.shuffle(neworder)
data = data[neworder,:]

print(data)

# Ten cross validation
training_length = len(data)/NB_CROSSVAL
KNNs = []
SSE = []
for i in xrange(NB_CROSSVAL):
	test_indices = range(training_length*i, training_length*(i+1))
	training_indices = [k for k in range(len(data)) if k not in test_indices]
	KNNs.append(KNN(training_indices))
	predictions = np.array([KNNs[i].predict(j) for j in test_indices])
	expected = np.array([data[j,10] for j in test_indices])
	SSE.append(sum((predictions - expected)**2))

print SSE
