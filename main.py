###############
# Lib import
###############

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import netCDF4
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
import time
import numpy as np
import cv2
from tqdm.auto import tqdm
import sys
import logging
import math

from random import random, randint
from numpy.random import choice

# Set tensorflow logger to avoid info message

mirrored_strategy = tf.distribute.MirroredStrategy()

if tf.__version__.startswith("1"):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
else:
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

###############
# Custom import
###############

from enums import enums
from Utils.utils import load_netcdf, load_path, normalize_img, denormalize_img, get_optimizer
from Network.network import Discriminator, Generator
from metrics import MSE, MAE

import argparse

# Combined network
def get_gan_network(discriminator, shape_lr, shape_hr, generator, optimizer):

    discriminator.trainable = False
    img_lr = Input(shape=shape_lr)
    fake_hr = generator(img_lr)
    
    gan_output = discriminator(fake_hr)
    
    out_hr = Lambda(lambda x: x, name='MSE')(fake_hr)
    out_gan = Lambda(lambda x: x, name='BCE')(gan_output)

    gan = Model(inputs=[img_lr], outputs=[out_gan, out_hr])

    gan.compile(loss=["binary_crossentropy", "mse"],
                loss_weights=[1e-3, 1],
                optimizer=optimizer)

    return gan

# Super Resolution Generative Adversarial Network (SRGAN)

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
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

glob_min_lr = [math.inf]
glob_max_lr = [-math.inf]
glob_min_hr = [math.inf]
glob_max_hr = [-math.inf]

# Get current timestamp and create directories
timestamp = str(time.time() + randint(1,100)).split('.')[0]
path = os.path.join('.', 'Experiments', timestamp)
if not os.path.exists(path):
    os.mkdir(path)
model_save_dir = os.path.join(path, 'models')

if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

logger.info('Saving to %s' % str(path))

if __name__ == "__main__":
    # Season argument
    parser.add_argument('--seasons', type=str, nargs='*', help='Define training season e.g. 01=Jan, 02=Feb')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=4, help='Define batch size')

    # Pretrain
    parser.add_argument('--pretrain', default=False, action='store_true', help='Bool type')

    args = parser.parse_args()

    # config
    train_test_ratio = enums.Config.TEST_TRAIN_RATIO.value
    epochs = enums.Config.EPOCHS.value
    batch_size = args.batch_size
    seasons = args.seasons
    pretrain = args.pretrain

    print('Current season: ', seasons)
    print('Pretrain: ', pretrain)

    ##############
    # Load dataset
    ##############
    files_lr_t2m, files_names_lr = load_netcdf(load_path(directory_lr), ext_in, variables_lr, channels, seasons)
    files_hr_t2m, files_names_hr = load_netcdf(load_path(directory_hr), ext_in, variables_lr, channels, seasons)
    # files_lr_t2m_test, files_names_lr_test = load_netcdf(load_path(directory_lr_test), ext_in, variables_lr, channels, seasons)
    # files_hr_t2m_test, files_names_hr_test = load_netcdf(load_path(directory_hr_test), ext_in, variables_lr, channels, seasons)

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

    # Compute global max and min channelwise
    for idx, i in enumerate(files_lr_t2m):
        glob_min_lr[0] = min(glob_min_lr[0], np.amin(i[:, :, 0]))
        glob_max_lr[0] = max(glob_max_lr[0], np.amax(i[:, :, 0]))

    # LR images normalization
    for idx, i in enumerate(files_lr_t2m):
        for c in range(channels):
            files_lr_t2m[idx, :, :, c] = normalize_img(i[:, :, c], lower_lr, upper_lr, glob_min_lr[0], glob_max_lr[0])

    # Compute global max and min channelwise
    for idx, i in enumerate(files_hr_t2m):
        glob_min_hr[0] = min(glob_min_hr[0], np.amin(i[:, :, 0]))
        glob_max_hr[0] = max(glob_max_hr[0], np.amax(i[:, :, 0]))

    # HR images normalization
    for idx, i in enumerate(files_hr_t2m):
        for c in range(channels):
            files_hr_t2m[idx, :, :, c] = normalize_img(files_hr_t2m[idx, :, :, c], lower_hr, upper_hr, glob_min_hr[0], glob_max_hr[0])

    logger.info('HR - Max: %f Min: %f' % (glob_max_hr[0], glob_min_hr[0]))
    logger.info('LR - Max: %f Min: %f' % (glob_max_lr[0], glob_min_lr[0]))

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

    del files_hr_pad
    del files_lr_pad
    del files_names_hr
    del files_names_lr

    # Train / Test splitting
    ## Total number of images
    number_of_images = len(files_lr_t2m)
    train_test_ratio = 0.972  # 205 images validation, 2%

    start_train_idx = 0
    end_train_idx = int(number_of_images * train_test_ratio)
    n_train_images = end_train_idx

    # Train/Val splitting
    files_lr_val = files_lr_t2m[end_train_idx:number_of_images, :, :, :]
    files_lr_t2m = files_lr_t2m[start_train_idx:end_train_idx, :, :, :]

    files_hr_val = files_hr_t2m[end_train_idx: number_of_images, :, :, :]
    files_hr_t2m = files_hr_t2m[start_train_idx: end_train_idx, :, :, :]


    logger.info("Shape LR files: %s" % str(files_lr_t2m[0].shape))
    logger.info("Shape HR files: %s" % str(files_hr_t2m[0].shape))
    logger.info("Shape LR VAL files: %s" % str(files_lr_val[0].shape))
    logger.info("Shape HR VAL files: %s" % str(files_hr_val[0].shape))

    ##########
    # Training
    ##########
    shape_image_hr = files_hr_t2m[0].shape
    shape_image_lr = files_lr_t2m[0].shape

    batch_count = int(files_hr_t2m.shape[0] / batch_size)
    optimizer = get_optimizer()

    with mirrored_strategy.scope():
        # With LR shape
        generator = Generator(shape_image_lr).generator()
        generator.compile(loss='mse', optimizer=optimizer)

        # With HR shape
        discriminator = Discriminator(shape_image_hr).discriminator()
        discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

        gan = get_gan_network(discriminator, shape_image_lr, shape_image_hr, generator, optimizer)

    loss_file = open(os.path.join(model_save_dir, 'losses.txt'), 'w+')
    loss_file.write('Training with batch_size %d, channels %d\n' % (batch_size, channels))
    loss_file.close()

    print(gan.summary())

    logger.info('Training with batch_size %d, channels %d' % (batch_size, channels))

    start_e = 1

    if pretrain == True:
        with mirrored_strategy.scope():
            # Pretrain only generator
            for e in range(1, 5):
                print('-' * 15, 'Epoch %d' % e, '-' * 15)

                start = time.time()

                for _ in tqdm(range(batch_count)):

                    rand_nums = np.random.randint(0, files_lr_t2m.shape[0], size=batch_size)
                    image_batch_lr = files_lr_t2m[rand_nums]
                    image_batch_hr = files_hr_t2m[rand_nums]

                    gan_loss = generator.train_on_batch(image_batch_lr, image_batch_hr)

                end = time.time()

                logger.info("Batch time (s): %d " % (end - start))
                logger.info("gan_loss : %s" % str(gan_loss))

                loss_file = open(os.path.join(model_save_dir, 'losses.txt'), 'a')
                loss_file.write('Epoch %d : gan_loss = %s ; time = %d\n' % (
                e, str(gan_loss), (end - start)))
                loss_file.close()

            gan = get_gan_network(discriminator, shape_image_lr, shape_image_hr, generator, optimizer)
            start_e = 6

    check = 0

    # Train GAN
    for e in range(start_e, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)

        start = time.time()

        for _ in tqdm(range(batch_count)):

            rand_nums = np.random.randint(0, files_lr_t2m.shape[0], size=batch_size)
            image_batch_lr = files_lr_t2m[rand_nums]
            image_batch_hr = files_hr_t2m[rand_nums]

            generated_images_sr = generator.predict(image_batch_lr)

            real = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake = np.random.random_sample(batch_size) * 0.2

            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real)

            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake)

            discriminator.trainable = False

            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            rand_nums = np.random.randint(0, files_lr_t2m.shape[0], size=batch_size)
            image_batch_lr = files_lr_t2m[rand_nums]
            image_batch_hr = files_hr_t2m[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2

            gan_loss = gan.train_on_batch(image_batch_lr, [gan_Y, image_batch_hr])

        end = time.time()


        logger.info("Batch time (s): %d " % (end-start))
        logger.info("gan_loss : %s" % str(gan_loss))
        logger.info("discriminator loss: %f" % discriminator_loss)

        loss_file = open(os.path.join(model_save_dir, 'losses.txt'), 'a')
        loss_file.write('Epoch %d : gan_loss = %s ; discriminator loss = %f, time = %d\n' % (e, str(gan_loss), discriminator_loss, (end-start) ) )
        loss_file.close()

        if e % enums.Config.CHECKPOINT_EPOCH.value == 0:
            generator.save(os.path.join(model_save_dir, 'gen_model_%d.h5' % e))
            discriminator.save(os.path.join(model_save_dir, 'dis_model_%d.h5' % e))

            mse_train = 0
            mse_val = 0
            for n_batches in range(64):

                rand_nums = np.random.randint(0, files_lr_t2m.shape[0], size=1)
                b_lr_train = files_lr_t2m[rand_nums]
                b_gen_train = generator.predict(b_lr_train)
                b_hr_train = files_hr_t2m[rand_nums]
                ## Remove padding to both hr and sr images
                normalized_generated = b_gen_train[0, 8:-9, 6:-7, 0]
                normalized_target = b_hr_train[0, 8:-9, 6:-7, 0]
                mse_train = mse_train + MSE(normalized_target, normalized_generated, None)

                rand_nums = np.random.randint(0, files_lr_val.shape[0], size=1)
                b_lr_val = files_lr_val[rand_nums]
                b_gen_val = generator.predict(b_lr_val)
                b_hr_val = files_hr_val[rand_nums]
                ## Remove padding to both hr and sr images
                normalized_generated = b_gen_val[0, 8:-9, 6:-7, 0]
                normalized_target = b_hr_val[0, 8:-9, 6:-7, 0]
                mse_val = mse_val + MSE(normalized_target, normalized_generated, None)

            mse_train = mse_train / 64
            mse_val = mse_val / 64

            if (check == 0):
                print("--------------------------")
                mse_file = open(os.path.join(path, 'mse_file.csv'), 'w')
                mse_file.write("epoch,mse_train,mse_val,ms_mse\n")
                mse_file.write('{},{},{},{}\n'.format(e, mse_train, mse_val, mse_train + abs(mse_train - mse_val)))
                mse_file.close()
            else:
                mse_file = open(os.path.join(path, 'mse_file.csv'), 'a')
                mse_file.write('{},{},{},{}\n'.format(e,mse_train,mse_val,mse_train+abs(mse_train-mse_val)))
                mse_file.close()

            check = 1

    sys.exit(0)

