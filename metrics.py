from scipy import stats
import time
import numpy as np
import cv2
import sys
import math
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import cv2
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from numpy.random import random
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from math import sqrt
import warnings


# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))


def log_spectral_distance(img1, img2):
    def power_spectrum_dB(img):
        fx = np.fft.fft2(img)
        fx = fx[:img.shape[0]//2,:img.shape[1]//2]
        px = abs(fx)**2
        return 10 * np.log10(px)

    d = (power_spectrum_dB(img1)-power_spectrum_dB(img2))**2

    d[~np.isfinite(d)] = np.nan
    return np.sqrt(np.nanmean(d))


def MSE(real, gen, axis):
    return np.mean((real - gen) ** 2, axis=axis)


def MAE(real, gen, axis):
    return np.mean(abs(real - gen), axis=axis)


def PSNR(real, gen, axis):
    MAX_PIX = np.amax(gen)
    mse = MSE(real, gen, axis=axis)
    return 10 * np.log10(MAX_PIX ** 2 / mse)


def SSIM_Array_2(real, gen):
    score = structural_similarity(real, gen, gaussian_weights=True, full=False, multichannel=True)
    return score


def SSIM_Array(real, gen):
    score = structural_similarity(real, gen, win_size=11, sigma=1.5, K1=0.01, K2=0.03,
                                  gaussian_weights=True, full=False, multichannel=False)
    return score


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = np.sqrt(sigma1.dot(sigma2))  # sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + sigma1 + sigma2 - 2.0 * covmean  # trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def FID(images1, images2):
    img1 = images1.reshape((1, images1.shape[0], images1.shape[1], 1))
    img2 = images2.reshape((1, images2.shape[0], images2.shape[1], 1))
    img1 = np.repeat(img1, 3, axis=3)
    img2 = np.repeat(img2, 3, axis=3)
    img1 = scale_images(img1, (299, 299, 3))
    img2 = scale_images(img2, (299, 299, 3))
    img1 = preprocess_input(img1)
    img2 = preprocess_input(img2)
    fid = calculate_fid(model, img1, img2)
    return fid


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)