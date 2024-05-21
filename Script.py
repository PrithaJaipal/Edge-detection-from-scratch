import cv2
import numpy as np
import os
from myEdgeFilter import myEdgeFilter

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the data and results directories relative to the script's location
datadir = os.path.join(script_dir, '../data')
resultsdir = os.path.join(script_dir, '../results1')

# Create the results directory if it does not exist
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)

# parameters
sigma     = 2
threshold = 0.07

# end of parameters

for file in os.listdir(datadir):
    if file.endswith('.jpg'):

        file = os.path.splitext(file)[0]
        
        # read in images
        img_path = os.path.join(datadir, f'{file}.jpg')
        img = cv2.imread(img_path)
        
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = np.float32(img) / 255
        
        # Edge detection and thresholding
        img_edge = myEdgeFilter(img, sigma)
        #img_edge = cv2.cvtColor(img_edge, cv2.COLOR_BGR2GRAY)

        img_threshold = np.float32(img_edge > threshold)

        # Save outputs (adjust file extensions if needed)
        fname = os.path.join(resultsdir, f'{file}_01edge.png')
        cv2.imwrite(fname, 255 * np.sqrt(img_edge / img_edge.max()))  # Assuming grayscale image

        fname = os.path.join(resultsdir, f'{file}_02threshold.png')
        cv2.imwrite(fname, 255 * img_threshold)