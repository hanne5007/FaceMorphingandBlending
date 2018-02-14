'''
  File name: getIndexes.py
  Author: Hantian Liu
  Date created:
'''

import numpy as np


def getIndexes(mask, targetH, targetW, offsetX, offsetY):
	[h, w] = np.shape(mask)
	x = np.count_nonzero(mask)
	# index the replacement points
	mask[mask == 1] = np.arange(x) + 1

	# filling to fit the size of target image
	indexes = np.column_stack((np.zeros([h, offsetX]), mask))
	if targetW - offsetX - w > 0:
		indexes = np.column_stack((indexes, np.zeros([h, targetW - offsetX - w])))
	indexes = np.row_stack((np.zeros([offsetY, targetW]), indexes))
	if targetH - offsetY - h > 0:
		indexes = np.row_stack((indexes, np.zeros([targetH - offsetY - h, targetW])))
	indexes = indexes[0:targetH, 0:targetW]

	return indexes
