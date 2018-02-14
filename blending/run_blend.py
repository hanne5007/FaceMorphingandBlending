#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/18/17 5:37 PM 

@author: Hantian Liu
"""

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
from seamlessCloningPoisson import seamlessCloningPoisson

simg = np.array(Image.open('source.jpg'))
timg = np.array(Image.open('target.jpg'))

# resize the image
[h,w,z]=np.shape(simg)
h=int(0.4*h)
w=int(0.4*w)
sim = scipy.misc.imresize(simg, [h, w])

offsetX=175
offsetY=520

'''
# sample image
simg = np.array(Image.open('SourceImage.jpg'))
timg = np.array(Image.open('TargetImage.jpg'))

[h,w,z]=np.shape(simg)
h=int(0.35*h)
w=int(0.35*w)
sim = scipy.misc.imresize(simg, [h, w])

offsetX=250
offsetY=180
'''

mask=maskImage(sim)

result=seamlessCloningPoisson(sim, timg, mask, offsetX, offsetY)

plt.figure(1)
plt.imshow(result)
plt.show()