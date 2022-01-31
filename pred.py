###############
# Lib import
###############

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pandas as pd
from scipy import stats
import netCDF4
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import time
import numpy as np
import cv2
from tqdm.auto import tqdm
import sys
import logging
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

warnings.filterwarnings("ignore")

# Set tensorflow logger to avoid info message

mirrored_strategy = tf.distribute.MirroredStrategy()

if tf.__version__.startswith("1"):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
else:
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

###############
# Custom import
###############

from enums import enums
from Utils.utils import load_netcdf, load_path, normalize_img, denormalize_img
from Network.network import Discriminator, Generator
from metrics import FID, log_spectral_distance, MSE, MAE, PSNR, SSIM_Array


# Super Resolution Generative Adversarial Network (SRGAN)


logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

channels = enums.Config.CHANNELS.value

directory_lr = enums.Dir.LR_PATH.value
directory_hr = enums.Dir.HR_PATH.value
directory_lr_test = enums.Dir.LR_PATH_TEST.value
directory_hr_test = enums.Dir.HR_PATH_TEST.value

# define netCDF4 data format
# TODO why HR has no T_2M value?
variables_lr = enums.Variables.VAR_LR.value
variables_hr = enums.Variables.VAR_HR.value

# define dats format ext
ext_in = enums.Format.EXT_IN.value
ext_out = enums.Format.EXT_OUT.value

# Normalize images
lower_lr = enums.Normalization.LOWER_LR.value
lower_hr = enums.Normalization.LOWER_HR.value
upper_lr = enums.Normalization.UPPER_LR.value
upper_hr = enums.Normalization.UPPER_HR.value

## Global Max and Min for each month
glob_min_lr = [ 211.113571, 212.097336, 214.805546, 219.644432, 231.490038, 240.254518, 247.434812, 237.902712, 231.078485, 221.021083, 216.611328, 211.868332 ]
glob_max_lr = [ 309.771243, 312.858591, 318.544744, 321.683902, 324.362793, 324.326643, 325.323975, 324.970947, 322.604809, 317.865924, 312.403801, 309.778196 ]
glob_min_hr = [ 210.671173, 211.506668, 214.134202, 219.644717, 231.376817, 240.049068, 245.740660, 237.516668, 230.672509, 220.903189, 216.611328, 210.889526 ]
glob_max_hr = [ 309.789537, 312.922346, 318.563762, 321.758068, 324.734863, 324.739514, 325.518311, 325.211914, 322.848271, 318.105018, 312.618137, 309.997491 ]

## Month-based models path
monthly_dir = ['1641436595', '1641503416', '1641575718', '1641943331', '1641922380', '1642717133', '1642779340', '1641466788',
               '1642778820', '1642974835', '1643006051', '1643007609']
epoch = [70, 60, 65, 75, 75, 75, 70, 70, 65, 50, 65, 95]

# DJF, MAM, JJA, SON path
seasonal_dir = {'DJF':'1643185246', 'MAM':'1643185178', 'JJA':'1643189168', 'SON':'1643189297'}
epoch_s = {'DJF':60, 'MAM':55, 'JJA':50, 'SON': 60}

if __name__ == "__main__":
    # Season argument
    parser.add_argument('--seasons', type=str, nargs='*', help='Define season e.g. 01=Jan, 02=Feb to get prediction'
                                                               'If seasaonal_model is not defined this serves as both season and month-based model')
    # Model
    parser.add_argument('--seasonal_model', type=str, default=None, help='Define seasonal model, e.g. DJF, MAM, JJA, SON')
    # Pretrain
    parser.add_argument('--inter_eval', default=False, action='store_true', help='Bool type')

    args = parser.parse_args()

    # config
    train_test_ratio = enums.Config.TEST_TRAIN_RATIO.value
    epochs = enums.Config.EPOCHS.value
    seasons = args.seasons
    mmodel = args.seasonal_model
    evaluation = args.inter_eval
    print(seasons, ", model", mmodel)

    ##############
    # Load dataset
    ##############

    files_lr_t2m, files_names_lr = load_netcdf(load_path(directory_lr_test), ext_in, variables_lr, channels, seasons)
    files_hr_t2m, files_names_hr = load_netcdf(load_path(directory_hr_test), ext_in, variables_lr, channels, seasons)
    # files_lr_t2m_test, files_names_lr_test = load_netcdf(load_path(directory_lr_test), ext_in, variables_lr, channels)
    # files_hr_t2m_test, files_names_hr_test = load_netcdf(load_path(directory_hr_test), ext_in, variables_lr, channels)

    if len(files_hr_t2m) == len(files_lr_t2m):
        logger.info("Length of input array is okay")
    else:
        logger.error("Shape inconsistency")
        sys.exit(1)

    logger.info("Number of images in LR training set: %d" % len(files_lr_t2m))
    logger.info("Number of images in HR training set: %d" % len(files_hr_t2m))
    # logger.info("Number of images in LR test set: %d" % len(files_lr_t2m_test))
    # logger.info("Number of images in HR test set: %d" % len(files_hr_t2m_test))

    # Change name format (?)
    files_names_lr = [name.strftime("%Y-%m-%d-%H:%M:%S") for name in files_names_lr]
    files_names_hr = [name.strftime("%Y-%m-%d-%H:%M:%S") for name in files_names_hr]
    # files_names_lr_test = [name.strftime("%Y-%m-%d-%H:%M:%S") for name in files_names_lr_test]
    # files_names_hr_test = [name.strftime("%Y-%m-%d-%H:%M:%S") for name in files_names_hr_test]

    mon_to_pick = int(seasons[0])
    ## Check if we're running in season- or month-based mode
    if mmodel == None:
        logger.info('Predicting with model: %d' % mon_to_pick)
        logger.info('EPOCH TO PICK: %d' % int(epoch[mon_to_pick-1]))
    else:
        logger.info('Predicting with %s' % mmodel)
        logger.info('EPOCH TO PICK: %d' % int(epoch_s[mmodel]))

    # Compute global max and min channelwise
    for idx, i in enumerate(files_lr_t2m):
        glob_min_lr[mon_to_pick-1] = min(glob_min_lr[mon_to_pick-1], np.amin(i[:, :, 0]))
        glob_max_lr[mon_to_pick-1] = max(glob_max_lr[mon_to_pick-1], np.amax(i[:, :, 0]))

    # LR images normalization
    for idx, i in enumerate(files_lr_t2m):
        for c in range(channels):
            files_lr_t2m[idx, :, :, c] = normalize_img(i[:, :, c], lower_lr, upper_lr, glob_min_lr[mon_to_pick-1], glob_max_lr[mon_to_pick-1])

    # Compute global max and min channelwise
    for idx, i in enumerate(files_hr_t2m):
        glob_min_hr[mon_to_pick-1] = min(glob_min_hr[mon_to_pick-1], np.amin(i[:, :, 0]))
        glob_max_hr[mon_to_pick-1] = max(glob_max_hr[mon_to_pick-1], np.amax(i[:, :, 0]))

    # HR images normalization
    for idx, i in enumerate(files_hr_t2m):
        for c in range(channels):
            files_hr_t2m[idx, :, :, c] = normalize_img(files_hr_t2m[idx, :, :, c], lower_hr, upper_hr, glob_min_hr[mon_to_pick-1],
                                                       glob_max_hr[mon_to_pick-1])

    logger.info('HR - Max: %f Min: %f' % (glob_max_hr[mon_to_pick-1], glob_min_hr[mon_to_pick-1]))
    logger.info('LR - Max: %f Min: %f' % (glob_max_lr[mon_to_pick-1], glob_min_lr[mon_to_pick-1]))

    ########
    # PAD LR
    ########
    # From 72 -> 112
    # From 158 -> 240
    logger.info('Padding LR images')
    files_lr_pad = np.zeros((files_lr_t2m.shape[0], 112, 240, channels))
    npad_lr = ((20, 20), (41, 41))
    for idx in tqdm(range(len(files_lr_t2m))):
        for c in range(channels):
            files_lr_pad[idx, :, :, c] = np.pad(files_lr_t2m[idx, :, :, c], pad_width=npad_lr, mode='edge')

    files_lr_t2m = files_lr_pad

    ########
    # PAD HR
    ########
    # 431 -> 448
    # 947 -> 960
    logger.info('Padding HR images')
    files_hr_pad = np.zeros((files_hr_t2m.shape[0], 448, 960, channels))
    npad_hr = ((8, 9), (6, 7))
    for idx in tqdm(range(len(files_hr_t2m))):
        for c in range(channels):
            files_hr_pad[idx, :, :, c] = np.pad(files_hr_t2m[idx, :, :, c], pad_width=npad_hr, mode='edge')

    files_hr_t2m = files_hr_pad

    logger.info("Shape LR files: %s" % str(files_lr_t2m[0].shape))
    logger.info("Shape HR files: %s" % str(files_hr_t2m[0].shape))

    ##########################
    # Inference and Evaluation
    #########################
    shape_image_hr = files_hr_t2m[0].shape
    shape_image_lr = files_lr_t2m[0].shape

    imgs_save_dir = 'images/12-Monthly'

    if not os.path.exists(imgs_save_dir):
        os.makedirs(imgs_save_dir)

    year_mask_2015 = [31] #31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    year_mask_2016 = [31] #31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    year_mask_2017 = [31] #31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    year_mask_2018 = [31] #31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    year_mask_test = np.r_[year_mask_2015, year_mask_2016, year_mask_2017, year_mask_2018]

    month_year_2015 = ["Dec-2015"] #, "Jan-2015", "Feb-2015", "Mar-2015", "Apr-2015", "May-2015", "Jun-2015", "Jul-2015", "Aug-2015", "Sep-2015", "Oct-2015", "Nov-2015", "Dec-2015" ]
    month_year_2016 = ["Dec-2016"] #, "Jan-2016", "Feb-2016", "Mar-2016", "Apr-2016", "May-2016", "Jun-2016", "Jul-2016", "Aug-2016", "Sep-2016", "Oct-2016", "Nov-2016", "Dec-2016" ]
    month_year_2017 = ["Dec-2017"] #, "Jan-2017", "Feb-2017", "Mar-2017", "Apr-2017", "May-2017", "Jun-2017", "Jul-2017", "Aug-2017", "Sep-2017", "Oct-2017", "Nov-2017", "Dec-2017" ]
    month_year_2018 = ["Dec-2018"] #, "Jan-2018", "Feb-2018", "Mar-2018", "Apr-2018", "May-2018", "Jun-2018", "Jul-2018", "Aug-2018", "Sep-2018", "Oct-2018", "Nov-2018", "Dec-2018" ]

    months_years = np.r_[month_year_2015, month_year_2016, month_year_2017, month_year_2018]

    with tf.device('/GPU:2'):

        if mmodel != None:
            # Seasonal
            generator = load_model('./Experiments/%s/models/gen_model_%d.h5' % ( seasonal_dir[mmodel], epoch_s[mmodel]), compile=False)
        else:
            # Monthly
            generator = load_model('./Experiments/%s/models/gen_model_%d.h5' % ( monthly_dir[mon_to_pick-1], epoch[mon_to_pick-1]), compile=False)
            print('Picked: ', mon_to_pick)
        # JFM - wrong
        # generator = load_model('./Experiments/1641577100/models/gen_model_60.h5', compile=False)

        lsd = 0.0
        ssim = 0
        psnr = 0
        mse = 0
        fid = 0
        img_count = 0

        month_days_cnt = 0
        cur_month = 0

        # hr image gathering
        image_batch_tp1_tmp = np.copy(files_hr_t2m[0][8:-9, 6:-7, :])
        image_batch_tp1_tmp[:, :, 0] = 0
        # generated sr images gathering
        generated_image_tmp = np.copy(files_hr_t2m[0][8:-9, 6:-7, :])
        generated_image_tmp[:, :, 0] = 0
        # rmse ?
        rmse_image_tmp = np.copy(files_hr_t2m[0][8:-9, 6:-7, :])
        rmse_image_tmp[:, :, 0] = 0
        # covariance
        covariance_tmp = np.copy(files_hr_t2m[0][8:-9, 6:-7, :])
        covariance_tmp[:, :, 0] = 0
        # coefficente di correlazione
        corrcoef_tmp = np.copy(files_hr_t2m[0][8:-9, 6:-7, :])
        corrcoef_tmp[:, :, 0] = 0
        # spearman tmp
        spearman_tmp = np.copy(files_hr_t2m[0][8:-9, 6:-7, :])
        spearman_tmp[:, :, 0] = 0
        # p value
        pvalue_tmp = np.copy(files_hr_t2m[0][8:-9, 6:-7, :])
        pvalue_tmp[:, :, 0] = 0
        # mappa di correlazione
        corr_tmp = np.copy(files_hr_t2m[0][8:-9, 6:-7, :])
        corr_tmp[:, :, 0] = 0

        image_batch_tp1_list = [ ]
        generated_image_list = [ ]


        for idx, i in tqdm(enumerate(files_lr_t2m)):

            '''
            if idx < 31*12:
                img_count = img_count + 1
                cur_month = 3
                continue
            '''

            month_days_cnt = month_days_cnt + 1
            img_count = img_count + 1

            start = time.time()
            img = generator.predict(i.reshape(1, i.shape[0], i.shape[1], i.shape[2]))
            end = time.time()

            n_generated = img[0, 8:-9, 6:-7, :]
            n_target = files_hr_t2m[idx, 8:-9, 6:-7, :]


            ### denormalize for maps
            d_generated = denormalize_img(img[0,:,:,:], lower_hr, upper_hr, glob_min_hr[0], glob_max_hr[0])[8:-9, 6:-7, :]
            d_target = denormalize_img(files_hr_t2m[idx,:,:,:] , lower_hr, upper_hr, glob_min_hr[0], glob_max_hr[0])[8:-9, 6:-7, :]

            # Kelvin to Celsius conv.
            target_img = d_target - 273.15
            gen_img = d_generated - 273.15

            if evaluation == True:

                # add gt image to tmp array and to list of generated images
                image_batch_tp1_tmp[:,:,0] += target_img[:,:,0]
                image_batch_tp1_list.append(target_img)
                # same for generated image
                generated_image_tmp[:,:,0] += gen_img[:,:,0]
                generated_image_list.append(gen_img[:,:,0])
                # add to rmse map
                rmse_image_tmp[:,:,0] += ((target_img[:,:,0] - gen_img[:,:,0]) ** 2)
                cur_step = year_mask_test[cur_month] * 4

                # If we reached one month
                if (month_days_cnt == cur_step):

                    for i in range(target_img.shape[0]):
                        for j in range(target_img.shape[1]):
                            dataset_real = [image_batch_tp1_list[k][i][j].item() for k in range(cur_step)]
                            dataset_gen = [generated_image_list[k][i][j] for k in range(cur_step)]
                            temp_dataset = np.array((dataset_real, dataset_gen))
                            covariance_tmp[i, j, 0] = np.cov(temp_dataset)[0, 1]
                            corrcoef_tmp[i, j, 0] = np.corrcoef(temp_dataset)[0, 1]
                            spearman = stats.spearmanr(temp_dataset, axis=1)
                            spearman_tmp[i, j, 0] = spearman[0]
                            pvalue_tmp[i, j, 0] = spearman[1]
                            corr_tmp[i, j, 0] = np.correlate(dataset_real, dataset_gen)

                    pd.DataFrame(image_batch_tp1_tmp[:,:,0] / cur_step).to_csv("{}/real_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False, index=False)
                    pd.DataFrame(generated_image_tmp[:,:,0] / cur_step).to_csv("{}/generated_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False, index=False)
                    pd.DataFrame( (image_batch_tp1_tmp[:,:,0] - generated_image_tmp[:,:,0]) / cur_step).to_csv("{}/diff_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False, index=False)
                    pd.DataFrame(np.abs(image_batch_tp1_tmp[:,:,0] - generated_image_tmp[:,:,0]) / cur_step).to_csv("{}/abs_diff_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False, index=False)
                    pd.DataFrame(np.sqrt(rmse_image_tmp[:,:,0] / cur_step)).to_csv("{}/rmse_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False, index=False)
                    pd.DataFrame(covariance_tmp[:,:,0]).to_csv("{}/covariance_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False,index=False)
                    pd.DataFrame(corrcoef_tmp[:,:,0]).to_csv("{}/corrcoef_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False,index=False)
                    pd.DataFrame(spearman_tmp[:,:,0]).to_csv("{}/spearman_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False,index=False)
                    pd.DataFrame(pvalue_tmp[2:,:,0]).to_csv("{}/pvalue_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False, index=False)
                    pd.DataFrame(corr_tmp[:,:,0]).to_csv("{}/corr_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header=False, index=False)
                    # reset
                    image_batch_tp1_list = []
                    generated_image_list = []

                    image_batch_tp1_tmp[:,:,0] = 0
                    generated_image_tmp[:,:,0] = 0
                    rmse_image_tmp[:,:,0] = 0
                    month_days_cnt = 0
                    cur_month += 1


            # Metrics gathering
            mse = mse + MSE(n_target, n_generated, axis=(0,1))
            psnr = psnr + PSNR(n_target, n_generated, axis=(0,1))
            lsd = lsd + log_spectral_distance(n_target[:,:,0], n_generated[:,:,0])
            fid = fid + FID(n_target[:, :, 0], n_generated[:, :, 0])
            ssim = ssim + SSIM_Array(n_target[:,:,0], n_generated[:,:,0])

        mse = mse / img_count
        psnr = psnr / img_count
        fid = fid / img_count
        ssim = ssim / img_count
        lsd = lsd / img_count
        fapp = ((1/mse)*psnr*ssim / (lsd*fid))

        logger.info('Avg time: %f' % (end - start))
        logger.info('MSE:%f - PSNR:%f - FID:%f - SSIM:%f - LSD:%f - 5Fapp:%f' % (mse, psnr, fid, ssim, lsd, fapp))

        logger.info('-' * 30)
