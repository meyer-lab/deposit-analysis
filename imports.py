import matplotlib.pyplot as plt


def load_images(tiff_file):
    im = plt.imread(tiff_file)
    return im
