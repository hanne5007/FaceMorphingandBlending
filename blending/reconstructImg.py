'''
  File name: reconstructImg.py
  Author: Hantian Liu
  Date created:
'''

from getCoefficientMatrix import getCoefficientMatrix
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def reconstructImg(indexes, red, green, blue, targetImg):
	coeffA = getCoefficientMatrix(indexes)

	# solve for x in Ax=b
	xred = scipy.sparse.linalg.spsolve(coeffA, red)
	xgreen = scipy.sparse.linalg.spsolve(coeffA, green)
	xblue = scipy.sparse.linalg.spsolve(coeffA, blue)
	'''
	np.clip(xred, 0, 255, out = xred)
	xred = xred.astype('uint8')
	np.clip(xgreen, 0, 255, out = xgreen)
	xgreen = xgreen.astype('uint8')
	np.clip(xblue, 0, 255, out = xblue)
	xblue = xblue.astype('uint8')
	'''

	# delete the original values of replacement points in target image
	ind0 = [indexes == 0]
	ind = [indexes != 0]
	resultImg = np.zeros(np.shape(targetImg))
	r = targetImg[:, :, 0] * ind0
	resultImg[:, :, 0] = r[0]
	g = targetImg[:, :, 1] * ind0
	resultImg[:, :, 1] = g[0]
	b = targetImg[:, :, 2] * ind0
	resultImg[:, :, 2] = b[0]

	# fill in the new values
	indexes[ind] = xred
	resultImg[:, :, 0] += indexes
	indexes[ind] = xgreen
	resultImg[:, :, 1] += indexes
	indexes[ind] = xblue
	resultImg[:, :, 2] += indexes

	# discretization
	np.clip(resultImg, 0, 255, out = resultImg)
	resultImg = resultImg.astype('uint8')

	return resultImg
