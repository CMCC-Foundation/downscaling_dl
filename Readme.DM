## Super Resolution GAN for downscaling 2-meters temperature

Directories:
*  ```Utils```: contains utilities, as data loader, normalization ecc..
* ```Enums```: enums.py builds set of global names as enumerations
* ```Network```:  contains network architecture scripts.
* ```Experiments```: directory to store experiments

Files:
* metrics.py contains relevant metrics functions (e.g. MSE, MAE ...).
* main.py; code for training the SR-GAN.
* pred.py: code to infer on test set.

### Script running

* Run ```python main.py --batch_size BATCH_SIZE --seasons SEASONS --pretrain``` to launch the training of the network. This script will store informations about both generator and discriminator in the Experiment directory. Furthermore, it computes validation metrics (i.e. MSE for both validation and train set)
    * BATCH_SIZE define the batch size, 
    * SEASONS define the months on which train the model; 
    * --pretrain tells to perform a pretraining of the generator.


* Run ```python pred.py --seasons SEASONS --seasonal_model SEM --inter_eval``` to perform inference on test set and generate interannual maps.
    * SEASONS define the month on which test the model; 
    * --seasonal_model: one between ['DJF', 'MAM', 'JJA', 'SON'], define which seasonal model to pick
    * --inter-eval tells to perform the intervaluation analysis (i.e. compute and store error maps and correlations) - if you run on the login node you should omit it. 

Paths on pred.py are hardcoded, you need to set the correct path to your model to get a prediction.

The evaluation maps has been plotted with the jupyter notebooks provided at: https://drive.google.com/drive/folders/1GenkyhZHDGxfTF2K12lw2-5dxbkjFLpW

All the metrics are computed on normalized fields.