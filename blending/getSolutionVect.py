'''
  File name: getSolutionVect.py
  Author: Hantian Liu
  Date created:
'''

import numpy as np
import scipy.signal


def getSolutionVect(indexes, source, target, offsetX, offsetY):
	# Laplacian on source image
	[m, n] = np.shape(source)
	d = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
	dsource = scipy.signal.convolve2d(source, d, boundary = 'fill', mode = 'same')

	# get values of replacements points in the b vector
	max_col = n + offsetX
	max_row = m + offsetY
	indexes_helper = indexes[offsetY + 1:max_row + 1, offsetX + 1:max_col + 1]
	ds = dsource
	ds = ds[indexes_helper != 0]

	[h, w] = np.shape(indexes)
	indexes = indexes.astype(np.int64)
	# number of replacement pixels
	N = np.count_nonzero(indexes)
	ind = indexes.nonzero()
	row = ind[0]
	col = ind[1]

	# search neighbors to add back the values for consistency in two boundaries
	add = np.zeros(np.shape(ds))
	# recursion on every replacement pixel
	for i in range(0, N):
		if col[i] + 1 < w and indexes[row[i], col[i] + 1] == 0:
			add[indexes[row[i], col[i]] - 1] += target[row[i], col[i] + 1]
		if row[i] + 1 < h and indexes[row[i] + 1, col[i]] == 0:
			add[indexes[row[i], col[i]] - 1] += target[row[i] + 1, col[i]]
		if col[i] - 1 >= 0 and indexes[row[i], col[i] - 1] == 0:
			add[indexes[row[i], col[i]] - 1] += target[row[i], col[i] - 1]
		if row[i] - 1 >= 0 and indexes[row[i] - 1, col[i]] == 0:
			add[indexes[row[i], col[i]] - 1] += target[row[i] - 1, col[i]]

	SolVectorb = ds + add

	return SolVectorb
