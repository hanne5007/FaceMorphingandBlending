#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/18/17 5:22 PM 

@author: Hantian Liu
"""

from __future__ import division
import numpy as np
from PIL import Image
from pylab import *
from click_correspondences import click_correspondences
from morph_tri import morph_tri
import scipy.misc
import imageio
import scipy.io as sio

img1 = np.array(Image.open('image1.jpg'))
img2 = np.array(Image.open('image2.jpg'))

# resize the images
im1 = scipy.misc.imresize(img1, [300, 300])
im2 = scipy.misc.imresize(img2, [300, 300])

im1_p = sio.loadmat('im1_pts.mat')
im1_pts = im1_p['im1_pts']
im2_p = sio.loadmat('im2_pts.mat')
im2_pts = im2_p['im2_pts']

#[im1_pts, im2_pts]=click_correspondences(im1, im2, 200)

# define warp and dissolve fraction
warp_frac=np.arange(0,1,1/50)
dissolve_frac=warp_frac

# morphing
morphed_im=morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)

# generate gifs
len=np.shape(warp_frac)
res_list=[]
for i in range(0,len[0]):
	res_list.append(morphed_im[i, :, :, :])
imageio.mimsave('./output.gif', res_list)



