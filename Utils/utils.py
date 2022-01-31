import pandas as pd
import numpy as np
from numpy.random import randint
import os
import netCDF4
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

def _valid_times(dataset, variable):
    '''Search dataset for time axis'''
    var = dataset.variables[variable]
    for d in var.dimensions:
        if d.startswith('time'):
            if d in dataset.variables:
                tvar = dataset.variables[d]
                return np.array(
                    netCDF4.num2date(tvar[:], units=tvar.units),
                    dtype='datetime64[s]')
    coords = var.coordinates.split()
    for c in coords:
        if c.startswith('time'):
            tvar = dataset.variables[c]
            return np.array(
                netCDF4.num2date(tvar[:], units=tvar.units),
                dtype='datetime64[s]')


def load_netcdf(dirs, ext, variable, channels, seasons, era = False):
    files = []
    names = []

    dirs = sorted(set(dirs))
    # For each dir in dirs
    for d in dirs:

        logger.info("Processing directory: %s " % d)
        count = 0
        sorted_listdir = sorted(os.listdir(d))
        for idx, f in enumerate(sorted_listdir):
            # if count > 3:
            #    break

            '''Check if the file is nc and January'''
            if f.endswith(ext) and (f.split('.')[0][-2:] in seasons):
                nc = netCDF4.Dataset(os.path.join(d, f))
                shp = nc.variables[variable].shape

                img = np.array(nc.variables[variable]).reshape(shp[0], shp[1], shp[2], channels)

                # TODO: ask for details. Why this value is calculated?
                '''if era:
                    offset = nc.variables[variable].add_offset
                    scaling_factor = nc.variables[variable].scale_factor
                    packed_array = (np.array(nc.variables[variable]) - offset)/scaling_factor

                    if -32767 in packed_array:
                        print('Missing value detected in LR dataset')
                else:
                    img = np.flip(img, axis = 1)'''

                if not count:
                    files = img
                else:
                    files = np.r_[files, img]

                count += 1
                # Retrieve names for each slice of the array
                names.extend(_valid_times(nc, variable).tolist())
                nc.close()

        logger.info("Loaded %s images count: %d" % (ext, count))

    return files, names


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)

    dirs = sorted(os.listdir(path))
    for elem in dirs:
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))

    return directories


def normalize_img(data, lower, upper, glob_min, glob_max):
    return ((upper-lower)*((data.astype(np.float32) - glob_min)/(glob_max - glob_min)) + lower)


def denormalize_img(data, lower, upper, glob_min, glob_max):
    return (((data.astype(np.float32) - lower)/(upper - lower))*(glob_max - glob_min) + glob_min)

def get_optimizer():
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam

# While training save generated image(in form LR, SR, HR)
# Save only one image as sample
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    logger.info("Plot generated images: %s" % str(examples))
    value = randint(0, examples)
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)

# ***
# While training save generated image(in form LR, SR, tp1)
# Save only one image as sample
def best_plot(generator, files_hr_pad_test, files_lr_2x2_test, glob_min, glob_max, channels, imgs_save_dir):

    examples = files_hr_pad_test.shape[0]
    
    time_cnt = 0.0
    psnr = 0.0
    mse = 0.0
    ssim = 0.0
    fid = 0.0
    month_days_cnt = 0
    cur_month = 0
    
    image_batch_tp1_tmp = np.copy(files_hr_pad_test[0])
    image_batch_tp1_tmp[:,:,0] = 0
    generated_image_tmp = np.copy(files_hr_pad_test[0])
    generated_image_tmp[:,:,0] = 0
    rmse_image_tmp = np.copy(files_hr_pad_test[0])
    rmse_image_tmp[:,:,0] = 0
    covariance_tmp = np.copy(files_hr_pad_test[0])
    covariance_tmp[:,:,0] = 0
    corrcoef_tmp = np.copy(files_hr_pad_test[0])
    corrcoef_tmp[:,:,0] = 0
    spearman_tmp = np.copy(files_hr_pad_test[0])
    spearman_tmp[:,:,0] = 0
    pvalue_tmp = np.copy(files_hr_pad_test[0])
    pvalue_tmp[:,:,0] = 0
    corr_tmp = np.copy(files_hr_pad_test[0])
    corr_tmp[:,:,0] = 0
    
    image_batch_tp1_list = [ ]
    generated_image_list = [ ]


    training_set_size = int(len(files_names_lr) - len(files_lr_80x160_test))
    examples = files_hr_pad_test.shape[0]
    
    for sample in range(examples):

        image_batch_t_resized = np.copy(files_lr_2x2_test[sample])
        image_batch_tp1 = np.copy(files_hr_pad_test[sample])
        

        for c in range(channels):
            image_batch_tp1[:, :, c] = denormalize_img(image_batch_tp1[:, :, c], lower_hr, upper_hr, glob_min[1][c], glob_max[1][c])
            image_batch_tp1_tmp[:,:,c] += image_batch_tp1[:,:,c]
            image_batch_tp1_list.append(image_batch_tp1)

        month_days_cnt += 1

        start_time = time.time( )
        generated_image = generator.predict(image_batch_t_resized.reshape(1, image_batch_t_resized.shape[0], image_batch_t_resized.shape[1], image_batch_t_resized.shape[2]))
        end_time = time.time()
   
        time_cnt += (end_time - start_time)
        generated_image_450x450 = generated_image[0]
        #print("GENERATED SHAPE : {}".format(generated_image_450x450.shape))

        #check_scales = ((e == 1) or (e == 10) or (e == 50) or ((e % scales_checkpoint_epoch) == 0))

        generated_image_450x450[0, :, :, c] = denormalize_img(generated_image[0][0, :, :, c], lower_hr, upper_hr, glob_min[1][c], glob_max[1][c])
        
        gen_image_dataframe = None
        
        if(outlier_removal):
            # OUTLIER REMOVAL procedure
            ###########################
            gen_image_dataframe = pd.DataFrame(generated_image_450x450[0, :, :, c])
            
            outlierConstant = 1
            upper_quartile = np.percentile(gen_image_dataframe, 75)
            lower_quartile = np.percentile(gen_image_dataframe, 25)
            IQR = (upper_quartile - lower_quartile) * outlierConstant
            quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

            gen_image_dataframe[(gen_image_dataframe < quartileSet[0])] = lower_quartile - IQR
            gen_image_dataframe[(gen_image_dataframe > quartileSet[0])] = lower_quartile + IQR

            generated_image_450x450[0,:,:,c] = gen_image_dataframe.values
            ###########################
        
        generated_image_tmp[:,:,c] += generated_image_450x450[0,:,:,c]
        
        generated_image_list.append(generated_image_450x450[0,:,:,c])
        
        rmse_image_tmp[:,:,c] += ((image_batch_tp1[:,:,c] - generated_image_450x450[0,:,:,c])**2)
        
        cur_step = year_mask_test[cur_month]*4
        
        if(month_days_cnt == cur_step):
      
            
            for c in range(channels):
                for i in range(image_batch_tp1.shape[0]):
                    for j in range(image_batch_tp1.shape[1]):
                        dataset_real = [image_batch_tp1_list[k][i][j].item() for k in range(cur_step)]
                        dataset_gen = [generated_image_list[k][i][j] for k in range(cur_step)]
                        temp_dataset = np.array((dataset_real, dataset_gen))
                        covariance_tmp[i,j,c] = np.cov(temp_dataset)[0,1]
                        corrcoef_tmp[i,j,c] = np.corrcoef(temp_dataset)[0,1]
                        spearman = stats.spearmanr(temp_dataset, axis=1)
                        spearman_tmp[i,j,c] = spearman[0]
                        pvalue_tmp[i,j,c] = spearman[1]
                        corr_tmp[i,j,c] = np.correlate(dataset_real, dataset_gen)
                        
            pd.DataFrame(image_batch_tp1_tmp[24:-25, 7:-6,c] / cur_step).to_csv("{}/real_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame(generated_image_tmp[24:-25, 7:-6,c] / cur_step).to_csv("{}/generated_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame((image_batch_tp1_tmp[24:-25, 7:-6,c]-generated_image_tmp[24:-25, 7:-6,c]) / cur_step).to_csv("{}/diff_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame(np.abs(image_batch_tp1_tmp[24:-25, 7:-6,c]-generated_image_tmp[24:-25, 7:-6,c]) / cur_step).to_csv("{}/abs_diff_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame(np.sqrt(rmse_image_tmp[24:-25, 7:-6,c]/cur_step)).to_csv("{}/rmse_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame(covariance_tmp[24:-25, 7:-6,c]).to_csv("{}/covariance_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame(corrcoef_tmp[24:-25, 7:-6,c]).to_csv("{}/corrcoef_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame(spearman_tmp[24:-25, 7:-6,c]).to_csv("{}/spearman_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame(pvalue_tmp[24:-25, 7:-6,c]).to_csv("{}/pvalue_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            pd.DataFrame(corr_tmp[24:-25, 7:-6,c]).to_csv("{}/corr_image_{}_T2M.csv".format(imgs_save_dir, months_years[cur_month]), header = False, index = False)
            
            image_batch_tp1_list = [ ]
            generated_image_list = [ ]
            
            image_batch_tp1_tmp[:,:,0] = 0
            generated_image_tmp[:,:,0] = 0
            rmse_image_tmp[:,:,0] = 0
            month_days_cnt = 0
            cur_month += 1
        
        #print("Denormalize GEN (t+1) : {} {}".format(min(np.ravel(generated_image)), max(np.ravel(generated_image))))

        #image_batch_t = denormalize_img(image_batch_t, 0, 1, glob_min[0], glob_max[0]) #denormalize_tp1(i)

        #print("Denormalize t : {} {}".format(min(np.ravel(image_batch_t)), max(np.ravel(image_batch_t))))

        mse += MSE(image_batch_tp1[24:-25, 7:-6,0], generated_image_450x450[0, 24:-25, 7:-6, 0], axis=(0,1))
        psnr += PSNR(image_batch_tp1[24:-25, 7:-6,0], generated_image_450x450[0, 24:-25, 7:-6, 0], axis=(0,1))
        ssim += SSIM_Array(image_batch_tp1[24:-25, 7:-6,0], generated_image_450x450[0, 24:-25, 7:-6, 0])
        fid += FID(image_batch_tp1[24:-25, 7:-6,0], generated_image_450x450[0, 24:-25, 7:-6, 0])
    
    mse /= examples
    psnr /= examples
    ssim /= examples
    fid /= examples
    print("MSE: {} ; PSNR: {} ; SSIM: {} ; FID: {}".format(mse, psnr, ssim, fid))
    return time_cnt

