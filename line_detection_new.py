import cv2
import os
import numpy as np
import math
import pandas as pd
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import probabilistic_hough_line
import skimage
from skimage import color
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu


def show_image(image, title='Image', cmap_type='gray'): 
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def threshold(image):
    thresh = threshold_otsu(gray)
    binary_global = gray > thresh
    return binary_global

def rescale_frame(frame, percent=20):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def count_levels(img):
    edges = canny(img, 0, 1, 200)
    n = int(img.shape[1]*0.5)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=n,
                                     line_gap=3)

    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, img.shape[1]))
    ax[2].set_ylim((img.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    return len(lines)/2
    
    