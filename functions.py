from PIL import Image, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor as exe

# Make a function that determines the value of the brightest pixel

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
    return cropped


# Make Gaussian blur of different images color coordinated
def blurring(images_folder):
    #images in list
    arr = os.listdir(images_folder)
    cropped = trim(arr, images_folder)
    length = len(cropped)

    # initialize vectors
    y_val = np.zeros(shape=(length, 51))

    e = exe()
    sigmas = range(y_val.shape[1])

    # iterate through list of images and run gaussian blurring
    for i in tqdm(range(0, len(cropped))):
        im = np.array(cropped[i], dtype=float)

        for j, maxval in zip(sigmas, e.map(lambda x: np.amax(gaussian_filter(im, x, mode="nearest")), sigmas)):
            y_val[i, j] = maxval

    return y_val


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
