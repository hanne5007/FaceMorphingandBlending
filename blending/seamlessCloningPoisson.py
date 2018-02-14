'''
  File name: seamlessCloningPoisson.py
  Author: Hantian Liu
  Date created:
'''

import numpy as np
from PIL import Image
from maskImage import maskImage
from getIndexes import getIndexes
import scipy.misc
import scipy.io as sio
from getCoefficientMatrix import getCoefficientMatrix
from getSolutionVect import getSolutionVect
from reconstructImg import reconstructImg
import matplotlib.pyplot as plt


def seamlessCloningPoisson(simg, timg, mask, offsetX, offsetY):
	[h, w, z] = np.shape(simg)
	[targetH, targetW, zz] = np.shape(timg)

	'''
	mask=maskImage(sim)
	print(mask)
	sio.savemat('mask.mat', {'mask':mask})

	'''
	# mask_load = sio.loadmat('mask')
	# mask = mask_load['mask']

	indexes = getIndexes(mask, targetH, targetW, offsetX, offsetY)

	red = getSolutionVect(indexes, simg[:, :, 0], timg[:, :, 0], offsetX, offsetY)
	green = getSolutionVect(indexes, simg[:, :, 1], timg[:, :, 1], offsetX, offsetY)
	blue = getSolutionVect(indexes, simg[:, :, 2], timg[:, :, 2], offsetX, offsetY)

	resultImg = reconstructImg(indexes, red, green, blue, timg)

	return resultImg
