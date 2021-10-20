import skimage
from skimage import filters, exposure
import os
from imports import load_images
from functions import brightest_pixel, trim
import matplotlib.pyplot as plt
import numpy as np


#images in list
arr = os.listdir('data/exp55')
cropped = trim(arr, 'data/exp55/')

# initialize vectors
im_blurred = [0] * 100
length = len(cropped)
y_val= np.zeros(shape = (length, 52))

# itterate through list of images and run gaussian blurring
for i in range(0, len(cropped)):
    im = np.array(cropped[i])

    for j in range(0, 51):
        sigma = j 
        im_blurred[j] = skimage.filters.gaussian(im, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
       
        #find brightest pixel in i
        y_val[i,j] = brightest_pixel(im_blurred[j])

# generate plot point for this sigma value
plt.figure('Exp 55')
plt.xlabel("Sigma")
plt.ylabel("Brightest Pixel")
plt.title('Experiment 55 ' + "Brightness")
plt.xlim([0,50])

for i in range(0,length):
        plt.plot(y_val[i], color = 'green')
    
print(y_val)
#plt.figure('averages')
plt.show()


