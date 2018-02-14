'''
  File name: morph_tri.py
  Author: Hantian Liu
  Date created: Fri Sep 29 11:30:54 2017
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
# from helpers import barycentric
import numpy as np
# from helpers import interp2
# import matplotlib.pyplot as plt
import math


def interpolate_points(x, y, im):
	size = np.shape(im)
	a = size[0] - 1
	b = size[1] - 1

	f_x = np.floor(x).astype(np.int64)
	c_x = np.ceil(x).astype(np.int64)
	f_x[f_x < 0] = 0
	c_x[c_x >= b] = b

	f_y = np.floor(y).astype(np.int64)
	c_y = np.ceil(y).astype(np.int64)
	f_y[f_y < 0] = 0
	c_y[c_y >= a] = a

	f_h = y - f_y
	c_h = 1 - f_h
	f_w = x - f_x
	c_w = 1 - f_w

	interp_val = im[f_x, f_y] * c_h * c_w + im[c_x, f_y] * c_h * f_w + im[f_x, c_y] * f_h * c_w + im[
																									  c_x, c_y] * f_h * f_w

	'''
    if f_x<0:
        f_x=0
    c_x = np.ceil(x).astype(np.int64)
    if c_x>b:
        c_x=b
    if f_y<0:
        f_y=0
    if c_y>a:
        c_y=a
    d1=math.sqrt((x - f_x) ** 2 + (y - f_y) ** 2)
    d2=math.sqrt((x-f_x)**2+(y-c_y)**2)
    d3=math.sqrt((x-c_x)**2+(y-c_y)**2)
    d4=math.sqrt((x-c_x)**2+(y-f_y)**2)

    d=d1+d2+d3+d4
    if d==0:
        interp_val=im[int(x),int(y)]
    else:
        interp_val=im[f_x,f_y]*d1/d+im[f_x,c_y]*d2/d+im[c_x,c_y]*d3/d+im[c_x,f_y]*d4/d
    '''
	return interp_val


def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
	# Tips: use Delaunay() function to get Delaunay triangulation;
	# Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.
	frame_num = np.size(warp_frac)
	imsize = np.shape(im1)
	im1_morphed = np.zeros([frame_num, imsize[0], imsize[1], imsize[2]])
	im2_morphed = np.zeros([frame_num, imsize[0], imsize[1], imsize[2]])
	morphed_im = np.zeros([frame_num, imsize[0], imsize[1], imsize[2]])

	# include boundary points
	im1_pts = np.row_stack((im1_pts, np.array([[0, 0], [0, 100], [0, 200], [0, 300], \
											   [100, 0], [200, 0], [300, 0], \
											   [300, 0], [300, 100], [300, 200], [300, 300], \
											   [100, 300], [200, 300]])))
	im2_pts = np.row_stack((im2_pts, np.array([[0, 0], [0, 100], [0, 200], [0, 300], \
											   [100, 0], [200, 0], [300, 0], \
											   [300, 0], [300, 100], [300, 200], [300, 300], \
											   [100, 300], [200, 300]])))

	for frame in range(0, frame_num):
		theta = warp_frac[frame]
		dis = dissolve_frac[frame]
		'''
        im_cur_pts=np.zeros(np.shape(im1_pts))
        im_cur_pts[:,1] = theta * im1_pts[:,0] + (1 - theta) * im2_pts[:,0]
        im_cur_pts[:,0] = theta * im1_pts[:,1] + (1 - theta) * im2_pts[:,1]
        '''
		im_cur_pts = (1 - theta) * im1_pts + theta * im2_pts

		tri = Delaunay(im_cur_pts)
		trisize = np.shape(tri.simplices)

		# find which triangle each [i,j] falls in
		fallsin = np.zeros([imsize[1], imsize[0]])
		for i in range(0, imsize[0]):
			for j in range(0, imsize[1]):
				fallsin[j, i] = tri.find_simplex(np.array([i, j]))

		for k in range(0, trisize[0]):
			# vertices of triangle k on the current image
			Ver = im_cur_pts[tri.simplices[k]]
			Ver = Ver.transpose()
			Ver = np.row_stack((Ver, np.ones([1, 3])))
			# pos of points which falls in the triangle k on the current image
			Pts = np.where(fallsin == k)
			# xnew = Pts[0]
			# ynew = Pts[1]
			xnew = Pts[1]
			ynew = Pts[0]
			Pts = np.row_stack((xnew, ynew))
			Pts = np.row_stack((Pts, np.ones([1, np.shape(Pts)[1]])))
			# alpha=[1,:], beta=[2,:], gamma=[3,:]
			bary_coor = np.dot(np.linalg.inv(Ver), Pts)

			# vertices of triangle k on image 1
			Ver_s1 = im1_pts[tri.simplices[k]]
			Ver_s1 = Ver_s1.transpose()
			Ver_s1 = np.row_stack((Ver_s1, np.ones([1, 3])))
			# pos of points which falls in the triangle k on image 1
			Pts_s1 = np.dot(Ver_s1, bary_coor)
			xs1 = Pts_s1[0:1, :] / Pts_s1[2:3, :]
			# xs1=Pts_s1[1:2, :] / Pts_s1[2:3, :]
			xs1 = xs1[0]
			ys1 = Pts_s1[1:2, :] / Pts_s1[2:3, :]
			# ys1=Pts_s1[0:1, :] / Pts_s1[2:3, :]
			ys1 = ys1[0]
			'''
            pts_num=np.shape(Pts)
            pts_num=pts_num[1]
            for i in range(0, pts_num):
                im1_morphed[frame, xnew[i],ynew[i],0]=interpolate_points(xs1[i],ys1[i],im1[:,:,0])
                im1_morphed[frame, xnew[i], ynew[i], 1] = interpolate_points(xs1[i], ys1[i], im1[:, :, 1])
                im1_morphed[frame, xnew[i], ynew[i], 2] = interpolate_points(xs1[i], ys1[i], im1[:, :, 2])
            
            im1_morphed[frame, xnew, ynew, 0] = interpolate_points(xs1, ys1, im1[:, :, 0])
            im1_morphed[frame, xnew, ynew, 1] = interpolate_points(xs1, ys1, im1[:, :, 1])
            im1_morphed[frame, xnew, ynew, 2] = interpolate_points(xs1, ys1, im1[:, :, 2])
            '''
			im1_morphed[frame, ynew, xnew, 0] = interpolate_points(ys1, xs1, im1[:, :, 0])
			im1_morphed[frame, ynew, xnew, 1] = interpolate_points(ys1, xs1, im1[:, :, 1])
			im1_morphed[frame, ynew, xnew, 2] = interpolate_points(ys1, xs1, im1[:, :, 2])

			# vertices of triangle k on image 2
			Ver_s2 = im2_pts[tri.simplices[k]]
			Ver_s2 = Ver_s2.transpose()
			Ver_s2 = np.row_stack((Ver_s2, np.ones([1, 3])))
			# pos of points which falls in the triangle k on image 2
			Pts_s2 = np.dot(Ver_s2, bary_coor)
			xs2 = Pts_s2[0:1, :] / Pts_s2[2:3, :]
			# xs2 = Pts_s2[1:2, :] / Pts_s2[2:3, :]
			xs2 = xs2[0]
			ys2 = Pts_s2[1:2, :] / Pts_s2[2:3, :]
			# ys2 = Pts_s2[0:1, :] / Pts_s2[2:3, :]
			ys2 = ys2[0]

			'''
            for i in range(0, pts_num):
                im2_morphed[frame, xnew[i], ynew[i], 0] = interpolate_points(xs2[i], ys2[i], im2[:, :, 0])
                im2_morphed[frame, xnew[i], ynew[i], 1] = interpolate_points(xs2[i], ys2[i], im2[:, :, 1])
                im2_morphed[frame, xnew[i], ynew[i], 2] = interpolate_points(xs2[i], ys2[i], im2[:, :, 2])
            
            im2_morphed[frame, xnew, ynew, 0] = interpolate_points(xs2, ys2, im2[:, :, 0])
            im2_morphed[frame, xnew, ynew, 1] = interpolate_points(xs2, ys2, im2[:, :, 1])
            im2_morphed[frame, xnew, ynew, 2] = interpolate_points(xs2, ys2, im2[:, :, 2])
            '''
			im2_morphed[frame, ynew, xnew, 0] = interpolate_points(ys2, xs2, im2[:, :, 0])
			im2_morphed[frame, ynew, xnew, 1] = interpolate_points(ys2, xs2, im2[:, :, 1])
			im2_morphed[frame, ynew, xnew, 2] = interpolate_points(ys2, xs2, im2[:, :, 2])

		# plt.figure(2*frame-1)
		# plt.imshow(im1_morphed[frame,:,:,:].astype('uint8'))
		# plt.figure(2*frame)
		# plt.imshow(im2_morphed[frame,:,:,:].astype('uint8'))
		morphed_im[frame, :, :, :] = (1 - dis) * im1_morphed[frame, :, :, :] + dis * im2_morphed[frame, :, :, :]

	# discretization
	np.clip(morphed_im, 0, 255, out = morphed_im)
	morphed_im = morphed_im.astype('uint8')

	return morphed_im
