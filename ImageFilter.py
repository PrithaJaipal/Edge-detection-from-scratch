import numpy as np

def myImageFilter(img0, h):
    
    #image dimensions
    img_size = img0.shape 
    img_height = img_size[0]
    img_width = img_size[1]

    #filter dimensions
    f_size = h.shape
    f_height = f_size[0]
    f_width = f_size[1]
    
    if f_height % 2 == 0 or f_width % 2 == 0:
        raise ValueError("Filter must be odd-sized.")
    
    #padding
    p_height = (f_height - 1) // 2
    p_width = (f_width - 1) // 2
    
    img_p = np.pad(img0, ((p_height, p_height), (p_width, p_width)), mode='edge')

    #convolution
    img1 = np.zeros_like(img0)

    for i in range(img_height):
        for j in range(img_width):
           img1[i, j] = np.sum(h * img_p[i:i+f_height, j:j+f_width])

    return img1


