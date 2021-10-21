from PIL import Image, ImageOps
from skimage.color import rgb2gray
import numpy as np
import sys
import skimage
from skimage import filters, exposure
import os
import matplotlib.pyplot as plt
import numpy as np

# Make a function that determines the value of the brightest pixel


def brightest_pixel(image):
    bw_image = rgb2gray(image)
    # Find brightest (max val)
    return np.amax(bw_image)


# make a function to trim the borders of images
def trim(arr, image_folder):
    image = [0] * len(arr)
    cropped = [0] * len(arr)
    for i in range(0, len(arr)):
        image[i] = Image.open(image_folder + arr[i])
        image[i].load()

        # remove alpha channel
        invert_im = image[i].convert("RGB")

        # invert image (so that white is 0)
        invert_im = ImageOps.invert(invert_im)
        imageBox = invert_im.getbbox()

        cropped[i] = image[i].crop(imageBox)
    return(cropped)

# Make Gaussian blur of different images color coordinated


def blurring(images_folder):
    #images in list
    arr = os.listdir(images_folder)
    cropped = trim(arr, images_folder)
    length = len(cropped)

    # initialize vectors
    #image1 = [0] * length1
    im_blurred = [0] * 100
    y_val = np.zeros(shape=(length, 52))

    # itterate through list of images and run gaussian blurring
    for i in range(0, len(cropped)):
        im = np.array(cropped[i])

        for j in range(0, 51):
            sigma = j
            im_blurred[j] = skimage.filters.gaussian(im, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

            # find brightest pixel in i
            # generate plot point for this sigma value
            y_val[i, j] = brightest_pixel(im_blurred[j])

    return(y_val)

# Create plot for image


def one_plot(y_val, experiment):
    plt.figure(experiment)
    plt.xlabel("Sigma")
    plt.ylabel("Brightest Pixel")
    plt.title("Image Brightness " + experiment)
    plt.xlim([0, 50])

    plt.plot(y_val[0], color='green', label=experiment)
    for i in range(1, len(y_val)):
        plt.plot(y_val[i], color='green')

    plt.legend()
    plt.show()


def compare_plots(images_folder1, images_folder2):
    #images in list
    arr1 = os.listdir(images_folder1)
    cropped1 = trim(arr1, images_folder1)
    length1 = len(cropped1)

    arr2 = os.listdir(images_folder2)
    cropped2 = trim(arr2, images_folder2)
    length2 = len(cropped2)

    # initialize vectors
    im_blurred1 = [0] * 100
    y_val1 = np.zeros(shape=(length1, 52))

    im_blurred2 = [0] * 100
    y_val2 = np.zeros(shape=(length2, 52))

    # itterate through list of images and run gaussian blurring
    for i in range(0, len(cropped1)):
        im1 = np.array(cropped1[i])

        for j in range(0, 51):
            sigma = j
            im_blurred1[j] = skimage.filters.gaussian(im1, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

            # find brightest pixel in i
            # generate plot point for this sigma value
            y_val1[i, j] = brightest_pixel(im_blurred1[j])

    for i in range(0, len(cropped2)):
        im2 = np.array(cropped2[i])

        for j in range(0, 51):
            sigma = j
            im_blurred2[j] = skimage.filters.gaussian(im2, sigma=(sigma, sigma), truncate=3.5, multichannel=True)

            # find brightest pixel in i
            # generate plot point for this sigma value
            y_val2[i, j] = brightest_pixel(im_blurred2[j])

    image_matrix = np.concatenate((y_val1, y_val2))
    return(image_matrix)
