'''
  File name: click_correspondences.py
  Author: Hantian Liu
  Date created: 
'''

'''
  File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

def click_correspondences(img1, img2, pts_num):
    '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
  '''

    im1 = scipy.misc.imresize(img1, [300, 300])
    im2 = scipy.misc.imresize(img2, [300, 300])

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(im1)
    plt.subplot(122)
    plt.imshow(im2)

    num = pts_num  # TODO
    x = plt.ginput(num, 0)
    x = np.array(x)
    im1_pts = x[0:num:2, :]
    im2_pts = x[1:num:2, :]

    plt.subplot(121)
    plt.plot(im1_pts[:, 0], im1_pts[:, 1], 'b+')
    plt.subplot(122)
    plt.plot(im2_pts[:, 0], im2_pts[:, 1], 'r+')
    plt.show()

    #print(im1_pts)
    #print(im2_pts)

    return im1_pts, im2_pts
