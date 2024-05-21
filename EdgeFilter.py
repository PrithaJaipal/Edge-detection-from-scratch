import numpy as np
from scipy import signal    # For signal.gaussian function 
from math import ceil
from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    # Assign sigma based on the filter size
    hsize = 2 * ceil(3 * sigma) + 1

    # Generate Gaussian signal and perform outer product to create 2D Gaussian filter
    h = np.outer(signal.windows.gaussian(hsize, sigma), signal.windows.gaussian(hsize, sigma))

    # Normalize filter so the filter values add up to 1
    h /= np.sum(h)

    # Smoothing
    img_s = myImageFilter(img0, h)

    # Define vertical and horizontal Sobel filters
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Image gradients in the x and y directions
    img_x = myImageFilter(img_s, sobel_x)
    img_y = myImageFilter(img_s, sobel_y)

    # Edge magnitude
    magn = np.sqrt(img_x**2 + img_y**2)

    # Edge orientation
    orientation = np.rad2deg(np.arctan2(img_y, img_x))
    orientation[orientation < 0] += 180  # Ensure angles are within [0, 180)

    # Non-maximum suppression
    N, M = magn.shape
    magn_s = np.zeros_like(magn)
    for i in range(1, N - 1):
        for j in range(1, M - 1):
            a = 255
            b = 255
            # Orientation is 0 degrees
            if (0 <= orientation[i, j] < 22.5) or (157.5 <= orientation[i, j] <= 180):
                a = magn[i, j+1]
                b = magn[i, j-1]
            # Orientation is 45 degrees
            elif (22.5 <= orientation[i, j] < 67.5):
                a = magn[i+1, j-1]
                b = magn[i-1, j+1]
            # Orientation is 90 degrees
            elif (67.5 <= orientation[i, j] < 112.5):
                a = magn[i+1, j]
                b = magn[i-1, j]
            # Orientation is 135 degrees
            elif (112.5 <= orientation[i, j] < 157.5):
                a = magn[i-1, j-1]
                b = magn[i+1, j+1]
            # Suppress non-max pixels
            if (magn[i, j] >= a) and (magn[i, j] >= b):
                magn_s[i, j] = magn[i, j]
    # return magn_s

    # Hysteresis thresholding to make edges continuous
    t_highRatio = 0.09
    t_lowRatio = 0.05

    t_high = np.max(magn_s) * t_highRatio
    t_low = t_high * t_lowRatio

    edges = np.zeros_like(magn_s)
    edges[magn_s >= t_high] = 255

    for i in range(1, N - 1):
         for j in range(1, M - 1):
             if (t_low <= magn_s[i, j] < t_high) and \
                ((magn_s[i-1:i+2, j-1:j+2] >= t_high).any()):
                 edges[i, j] = 255

    return edges
