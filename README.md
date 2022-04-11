
# astromlp-models

Deep learning models for astrophysics applications.

## Topic: Galaxies

Collection of Deep Learning models to characterize different proprieties of galaxies,
based on data from the [Sloan Digital Sky Survey](https://www.sdss.org/) (SDSS).

The [SDSS Galaxy Subset](https://zenodo.org/record/6393488) dataset is used to
train and explore available models (the default location of the dataset w.r.t. to this repository
is `../sdss-gs`).

### Models

The following table quickly describes single input/output models available.

|  Model   |  Input       |  Output        |  Type        |  Description      |
|:--------:|:------------:|:--------------:|:------------:|-------------------|
| `i2r`    |  `img`       |  `redshift`    |  regression  |  infer object redshift from RGB image  |
| `f2r`    |  `fits`      |  `redshift`    |  regression  |  infer object redshift from FITS data  |
| `s2r`    |  `spectra`   |  `redshift`    |  regression  |  infer object redshift from spectra data  |
| `ss2r`   |  `ssel`      |  `redshift`    |  regression  |  infer object redshift from selected spectra data  |
| `b2r`    |  `bands`     |  `redshift`    |  regression  |  infer object redshift from bands data  |
| `w2r`    |  `wise`      |  `redshift`    |  regression  |  infer object redshift from WISE data  |
| `i2sm`   |  `img`       |  `smass`       |  regression  |  infer object stellar mass from RGB image  |
| `f2sm`   |  `fits`      |  `smass`       |  regression  |  infer object stellar mass from FITS data  |
| `s2sm`   |  `spectra`   |  `smass`       |  regression  |  infer object stellar mass from spectra data  |
| `ss2sm`  |  `ssel`      |  `smass`       |  regression  |  infer object stellar mass from selected spectra data  |
| `b2sm`   |  `bands`     |  `smass`       |  regression  |  infer object stellar mass from bands data  |
| `w2sm`   |  `wise`      |  `smass`       |  regression  |  infer object stellar mass from WISE data  |
| `i2s`    |  `img`       |  `subclass`    |  classification  |  infer object sub-class from RGB image  |
| `f2s`    |  `fits`      |  `subclass`    |  classification  |  infer object sub-class from FITS data  |
| `s2s`    |  `spectra`   |  `subclass`    |  classification  |  infer object sub-class from spectra data  |
| `ss2s`   |  `ssel`      |  `subclass`    |  classification  |  infer object sub-class from selected spectra data  |
| `b2s`    |  `bands`     |  `subclass`    |  classification  |  infer object sub-class from bands data  |
| `w2s`    |  `wise`      |  `subclass`    |  classification  |  infer object sub-class from WISE data  |
| `i2g`    |  `img`       |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from RGB image  |
| `f2g`    |  `fits`      |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from FITS data  |
| `s2g`    |  `spectra`   |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from spectra data  |
| `ss2g`   |  `ssel`      |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from selected spectra data  |
| `b2g`    |  `bands`     |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from bands data  |
| `w2g`   |  `wise`      |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from WISE data  |

The following table quickly summarizes the multi-input/output models available.

|  Model             |  Input(s)                                 |  Output(s)                            |
|:------------------:|:-----------------------------------------:|:-------------------------------------:|
| `fSbW2rSM`         |  `fits, spectra, bands, wise`             |  `redshift, smass`                    |
| `fSbW2sG`          |  `fits, spectra, bands, wise`             |  `subclass, gz2c`                     |
| `iFsSSbW2r`        |  `img, fits, spectra, ssel, bands, wise`  |  `redshift`                           |
| `iFsSSbW2sm`       |  `img, fits, spectra, ssel, bands, wise`  |  `smass`                              |
| `iFsSSbW2s`        |  `img, fits, spectra, ssel, bands, wise`  |  `subclass`                           |
| `iFsSSbW2g`        |  `img, fits, spectra, ssel, bands, wise`  |  `gz2c`                               |
| `iFsSSbW2rSMsG`    |  `img, fits, spectra, ssel, bands, wise`  |  `redshift, smass, subclass, gz2c`    |



### Inputs & Outputs

The following tables describe the data used as inputs for the models, and the
outputs for regression and classification models.

|  Input      |  Shape      |  Description  |
|:-----------:|:-----------:|---------------|
|  `img`      |  150x150x3  |  RGB image from the object in JPEG format, 150x150 pixels, generated using the [SkyServer DR16 API](http://skyserver.sdss.org/dr16/en/help/docs/api.aspx)  |
|  `fits`     |  61x61x5    |  FITS data subset around the object across the u, g, r, i, z bands; cut is done using the [ImageCutter](https://github.com/jhoar/ImageCutter) library  |
|  `spectra`  |  3225x1     |  full best fit spectra data from SDSS between 4000 and 9000 wavelengths  |
|  `ssel`     |  1225x1     |  best fit spectra data from SDSS for specific selected intervals of wavelengths discussed by [Sánchez Almeida 2010](https://arxiv.org/abs/1003.3186)  |
|  `bands`    |  5x1        |  photometric values from SDSS data: `[modelMag_u, modelMag_g, modelMag_r, modelMag_i, modelMag_z]` ([data table details](http://skyserver.sdss.org/dr16/en/help/browser/browser.aspx#&&history=description+SpecPhoto+V))  |
|  `wise`     |  4x1        |  list of bands values from WISE data: `[w1mag, w2mag, w3mag, w4mag]` ([data table details](http://skyserver.sdss.org/dr16/en/help/browser/browser.aspx#&&history=description+WISE_allsky+U))  |

|  Output      |  Type        |  Description  |
|:------------:|:------------:|---------------|
|  `redshift`  |  regression  |  final redshift from SDSS data `z` ([data table details](http://skyserver.sdss.org/dr16/en/help/browser/browser.aspx#&&history=description+SpecPhoto+V))  |
|  `subclass`  |  classification  | subset of sub-class from SDSS data for the galaxy objects `subClass` ([data table details](http://skyserver.sdss.org/dr16/en/help/browser/browser.aspx#&&history=description+SpecPhoto+V))  |
|  `smass`     |  regression   |  stellar mass extracted from the [eBOSS Firefly catalog](https://www.sdss.org/dr16/spectro/eboss-firefly-value-added-catalog)  |
|  `gz2c`      |  classification  | simplified version of the Galaxy Zoo 2 classification, from [Willett et al 2013](https://academic.oup.com/mnras/article/435/4/2835/1022913) (see class sets section below for details)  |

### Class Sets

The following tables describe the class set labels for the classification outputs.

#### SDSS sub-class subset (`subclass`)

The sub-class parameter for each object is available from the
[SDSS spectroscopic catalogs](https://www.sdss.org/dr17/spectro/catalogs/).

|  Label        |  Description  |
|:-------------:|---------------|
|  `AGN`        |  has detectable emission lines that are consistent with being a Seyfert or LINER  |
|  `BROADLINE`  |  has lines detected at the 10-sigma level with sigmas > 200 km/sec at the 5-sigma level  |
|  `STARBURST`  |  galaxy is star-forming  |
|  `STARFORMING`|  has detectable emission lines that are consistent with star-formation criteria  |

#### Galaxy Zoo 2 Simplified Classes (`gz2c`)

This is a simplified version of the Galaxy Zoo 2 classification tree,
detailed information on the original data is available [here](https://data.galaxyzoo.org/).

|  Label     |  Description  |
|:----------:|---------------|
|  `A`       |  artifact, star  |
|  `Ec`      |  smooth, cigar-shaped  |
|  `Ei`      |  smooth, in-between  |
|  `Er`      |  smooth, completely round  |
|  `SBa`     |  with features/disks, has bar, dominant bulge prominence  |
|  `SBaR`    |  with features/disks, has bar, dominant bulge prominence, has spiral structure  |
|  `SBb`     |  with features/disks, has bar, obvious bulge prominence  |
|  `SBbR`    |  with features/disks, has bar, obvious bulge prominence, has spiral structure  |
|  `SBc`     |  with features/disks, has bar, just noticeable bulge prominence  |
|  `SBcR`    |  with features/disks, has bar, just noticeable bulge prominence, has spiral structure  |
|  `SBd`     |  with features/disks, has bar, no bulge prominence  |
|  `SBdR`    |  with features/disks, has bar, no bulge prominence, has spiral structure  |
|  `Sa`      |  with features/disks, dominant bulge prominence  |
|  `SaR`     |  with features/disks, dominant bulge prominence, has spiral structure  |
|  `Sb`      |  with features/disks, obvious bulge prominence  |
|  `SbR`     |  with features/disks, obvious bulge prominence, has spiral structure  |
|  `Sc`      |  with features/disks, just noticeable bulge prominence  |
|  `ScR`     |  with features/disks, just noticeable bulge prominence, has spiral structure  |
|  `Sd`      |  with features/disks, no bulge prominence  |
|  `SdR`     |  with features/disks, no bulge prominence, has spiral structure  |
|  `Seb`     |  with features/disks, edge-on, boxy bulge  |
|  `Sen`     |  with features/disks, edge-on, no bulge  |
|  `Ser`     |  with features/disks, edge-on, round bulge  |


### Fitting Models and Visualizing Metrics

The models available in this repository are implemented using [Keras](https://keras.io/).
To fit the models available in this repository the [astromlp](https://github.com/nunorc/astromlp)
Python package that includes all the helper classes is also required.

You can fit a model using [mlflow](https://mlflow.org/), for example to fit the `i2r` model using
your current `python` (i.e. don't create a new environment using `conda`) you can run from
the repository directory, and also include this run in the `i2r` experiment:

    $ mlflow run i2r --experiment-name i2r --no-conda

You can also change the parameters to run the model, namely the number of epochs, the batch size,
the loss function and optimizer to use, for example:

    $ mlflow run i2r -P epochs=10 -P batch_size=32 -P loss=mse -P optimizer=adam --experiment-name i2r --no-conda

You can also change the location of the dataset by setting the `ds` parameter:

    $ mlflow run i2r -P ds=/tmp/sdss-gs --experiment-name i2r --no-conda

To view the data concerning the fitting of the available models you can use `mlflow` user interface:

    $ mlflow ui --backend-store-uri sqlite:///mlruns.db

To fit models and include data in this database you can set the `MLFLOW_TRACKING_URI` environment
variable to this file, remember to use an absolute path, for example:

    $ export MLFLOW_TRACKING_URI=sqlite:////home/nrc/astromlp-models.git/mlruns.db

And can also check the generated [tensorboard](https://www.tensorflow.org/tensorboard) logs, for
example:

    $ tensorboard --logdir i2r/logs/

### Single Input/Output Models Hyper-parameters Exploration

The following table summarizes different combinations of batch size (32 or 64),
optimizer (RMSProp or Adam) and loss functions (Mean Squared Error or Mean Absolute
Error for regression, categorical crossentropy is always used for multi-label
classification) hyper-parameters exploration, the score is the best validation value for the
validation dataset. The highlighted row describes the final hyper-parameters used for
bootstrapping each model and the corresponding evaluation score (accuracy for classification
models) on the test set, never seen by any model during the exploration steps.

|  Model    |  Input      |  Output        |  Epochs  |  Batch Size  |  Optimizer   |  Loss     |   Score    |
|:---------:|:-----------:|:--------------:|:--------:|:------------:|:------------:|:---------:|:----------:|
|  `i2r`    |  `img`      |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.003365  |
|  `i2r`    |  `img`      |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.030170  |
|  `i2r`    |  `img`      |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.016661  |
|  `i2r`    |  `img`      |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.071612  |
|  `i2r`    |  `img`      |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.003508  |
|  `i2r`    |  `img`      |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.030480  |
|  `i2r`    |  `img`      |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.003498  |
|  `i2r`    |  `img`      |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.071558  |
|**`i2r`**  |**`img`**    |**`redshift`**  |  **20**  |  **32**      |  **RMSProp** |  **MSE**  |**0.002377**|
|  `f2r`    |  `fits`     |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.001930  |
|  `f2r`    |  `fits`     |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.027434  |
|  `f2r`    |  `fits`     |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.001917  |
|  `f2r`    |  `fits`     |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.024896  |
|  `f2r`    |  `fits`     |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.002075  |
|  `f2r`    |  `fits`     |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.028206  |
|  `f2r`    |  `fits`     |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.001870  |
|  `f2r`    |  `fits`     |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.024997  |
|**`f2r`**  |**`fits`**   |**`redshift`**  |  **20**  |  **64**      |  **Adam**    |  **MSE**  |**0.002020**|
|  `s2r`    |  `spectra`  |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.004649  |
|  `s2r`    |  `spectra`  |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.033334  |
|  `s2r`    |  `spectra`  |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.006617  |
|  `s2r`    |  `spectra`  |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.040978  |
|  `s2r`    |  `spectra`  |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.005554  |
|  `s2r`    |  `spectra`  |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.038496  |
|  `s2r`    |  `spectra`  |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.014275  |
|  `s2r`    |  `spectra`  |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.044180  |
|**`s2r`**  |**`spectra`**|**`redshift`**  |  **20**  |  **32**      |  **RMSProp** |  **MSE**  |**0.004401**|
|  `ss2r`   |  `ssel`     |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.004343  |
|  `ss2r`   |  `ssel`     |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.030403  |
|  `ss2r`   |  `ssel`     |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.008098  |
|  `ss2r`   |  `ssel`     |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.045030  |
|  `ss2r`   |  `ssel`     |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.005174  |
|  `ss2r`   |  `ssel`     |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.031783  |
|  `ss2r`   |  `ssel`     |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.006342  |
|  `ss2r`   |  `ssel`     |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.053812  |
|**`ss2r`** |**`ssel`**   |**`redshift`**  |  **20**  |  **32**      |  **RMSProp** |  **MSE**  |**0.003700**|
|  `b2r`    |  `bands`    |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.003765  |
|  `b2r`    |  `bands`    |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.027912  |
|  `b2r`    |  `bands`    |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.004437  |
|  `b2r`    |  `bands`    |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.027528  |
|  `b2r`    |  `bands`    |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.004726  |
|  `b2r`    |  `bands`    |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.029070  |
|  `b2r`    |  `bands`    |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.004593  |
|  `b2r`    |  `bands`    |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.027921  |
|**`b2r`**  |**`bands`**  |**`redshift`**  |  **20**  |  **32**      |  **RMSProp** |  **MSE**  |**0.002806**|
|  `w2r`    |  `wise`     |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.011475  |
|  `w2r`    |  `wise`     |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.055713  |
|  `w2r`    |  `wise`     |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.011523  |
|  `w2r`    |  `wise`     |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.055016  |
|  `w2r`    |  `wise`     |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.011657  |
|  `w2r`    |  `wise`     |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.056263  |
|  `w2r`    |  `wise`     |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.011557  |
|  `w2r`    |  `wise`     |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.055382  |
|**`w2r`**  |**`wise`**   |**`redshift`**  | **20**   |  **32**      |  **RMSProp** |  **MSE**  |**0.012004**|
|  `i2sm`   |  `img`      |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  40929.859 |
|  `i2sm`   |  `img`      |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  26.048    |
|  `i2sm`   |  `img`      |  `smass`       |  10      |  32          |  Adam        |  MSE      |  40813.363 |
|  `i2sm`   |  `img`      |  `smass`       |  10      |  32          |  Adam        |  MAE      |  26.695    |
|  `i2sm`   |  `img`      |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  40982.949 |
|  `i2sm`   |  `img`      |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  26.736    |
|  `i2sm`   |  `img`      |  `smass`       |  10      |  64          |  Adam        |  MSE      |  42835.566 |
|  `i2sm`   |  `img`      |  `smass`       |  10      |  64          |  Adam        |  MAE      |  26.238    |
|**`i2sm`** |**`img`**    |**`smass`**     |  **20**  |  **64**      |  **Adam**    |  **MAE**  |**22.537**  |
|  `f2sm`   |  `fits`     |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  40600.176 |
|  `f2sm`   |  `fits`     |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  27.171    |
|  `f2sm`   |  `fits`     |  `smass`       |  10      |  32          |  Adam        |  MSE      |  40491.633 |
|  `f2sm`   |  `fits`     |  `smass`       |  10      |  32          |  Adam        |  MAE      |  25.758    |
|  `f2sm`   |  `fits`     |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  40679.352 |
|  `f2sm`   |  `fits`     |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  26.600    |
|  `f2sm`   |  `fits`     |  `smass`       |  10      |  64          |  Adam        |  MSE      |  40903.289 |
|  `f2sm`   |  `fits`     |  `smass`       |  10      |  64          |  Adam        |  MAE      |  25.931    |
|**`f2sm`** |**`fits`**   |**`smass`**     |  **20**  |  **32**      |  **Adam**    |  **MAE**  |**21.753**  |
|  `s2sm`   |  `spectra`  |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  2963.335  |
|  `s2sm`   |  `spectra`  |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  20.871    |
|  `s2sm`   |  `spectra`  |  `smass`       |  10      |  32          |  Adam        |  MSE      |  3432.733  |
|  `s2sm`   |  `spectra`  |  `smass`       |  10      |  32          |  Adam        |  MAE      |  20.129    |
|  `s2sm`   |  `spectra`  |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  3159.474  |
|  `s2sm`   |  `spectra`  |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  21.317    |
|  `s2sm`   |  `spectra`  |  `smass`       |  10      |  64          |  Adam        |  MSE      |  3518.646  |
|  `s2sm`   |  `spectra`  |  `smass`       |  10      |  64          |  Adam        |  MAE      |  20.420    |
|**`s2sm`** |**`spectra`**|**`smass`**     |  **20**  |  **32**      |  **Adam**    |  **MAE**  |**19.577**  |
|  `ss2sm`  |  `ssel`     |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  40206.445 |
|  `ss2sm`  |  `ssel`     |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  22.797    |
|  `ss2sm`  |  `ssel`     |  `smass`       |  10      |  32          |  Adam        |  MSE      |  3018.159  |
|  `ss2sm`  |  `ssel`     |  `smass`       |  10      |  32          |  Adam        |  MAE      |  22.728    |
|  `ss2sm`  |  `ssel`     |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  40271.656 |
|  `ss2sm`  |  `ssel`     |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  24.099    |
|  `ss2sm`  |  `ssel`     |  `smass`       |  10      |  64          |  Adam        |  MSE      |  40798.273 |
|  `ss2sm`  |  `ssel`     |  `smass`       |  10      |  64          |  Adam        |  MAE      |  22.380    |
|**`ss2sm`**|**`ssel`**   |**`smass`**     |  **20**  |  **64**      |  **Adam**    |  **MAE**  |**18.793**  |
|  `b2sm`   |  `bands`    |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  3387.860  |
|  `b2sm`   |  `bands`    |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  26.523    |
|  `b2sm`   |  `bands`    |  `smass`       |  10      |  32          |  Adam        |  MSE      |  5747.023  |
|  `b2sm`   |  `bands`    |  `smass`       |  10      |  32          |  Adam        |  MAE      |  25.930    |
|  `b2sm`   |  `bands`    |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  5520.346  |
|  `b2sm`   |  `bands`    |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  26.694    |
|  `b2sm`   |  `bands`    |  `smass`       |  10      |  64          |  Adam        |  MSE      |  5725.097  |
|  `b2sm`   |  `bands`    |  `smass`       |  10      |  64          |  Adam        |  MAE      |  25.151    |
|**`b2sm`** |**`bands`**  |**`smass`**     |  **20**  |  **64**      |  **Adam**    |  **MAE**  |**25.344**  |
|  `w2sm`   |  `wise`     |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  4323.260  |
|  `w2sm`   |  `wise`     |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  30.034    |
|  `w2sm`   |  `wise`     |  `smass`       |  10      |  32          |  Adam        |  MSE      |  4391.042  |
|  `w2sm`   |  `wise`     |  `smass`       |  10      |  32          |  Adam        |  MAE      |  29.858    |
|  `w2sm`   |  `wise`     |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  4423.225  |
|  `w2sm`   |  `wise`     |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  30.605    |
|  `w2sm`   |  `wise`     |  `smass`       |  10      |  64          |  Adam        |  MSE      |  4433.951  |
|  `w2sm`   |  `wise`     |  `smass`       |  10      |  64          |  Adam        |  MAE      |  29.800    |
|**`w2sm`** |**`wise`**   |**`smass`**     |  **20**  |  **64**      |  **Adam**    |  **MAE**  |**27.719**  |
|  `i2s`    |  `img`      |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.771606  |
|  `i2s`    |  `img`      |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.778097  |
|  `i2s`    |  `img`      |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.767020  |
|  `i2s`    |  `img`      |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.787760  |
|**`i2s`**  |**`img`**    |**`subclass`**  |  **20**  |  **64**      |  **Adam**    |  **categorical crossentropy**  |**0.765625**|
|  `f2s`    |  `fits`     |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.771977  |
|  `f2s`    |  `fits`     |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.782177  |
|  `f2s`    |  `fits`     |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.766183  |
|  `f2s`    |  `fits`     |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.778739  |
|**`f2s`**  |**`fits`**   |**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.785590**|
|  `s2s`    |  `spectra`  |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.765579  |
|  `s2s`    |  `spectra`  |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.767897  |
|  `s2s`    |  `spectra`  |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.766648  |
|  `s2s`    |  `spectra`  |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.713914  |
|**`s2s`**  |**`spectra`**|**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.763455**|
|  `ss2s`   |  `ssel`     |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.765764  |
|  `ss2s`   |  `ssel`     |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.768824  |
|  `ss2s`   |  `ssel`     |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.761719  |
|  `ss2s`   |  `ssel`     |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.768415  |
|**`ss2s`** |**`ssel`**   |**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.756076**|
|  `b2s`    |  `bands`    |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.758995  |
|  `b2s`    |  `bands`    |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.762148  |
|  `b2s`    |  `bands`    |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.757720  |
|  `b2s`    |  `bands`    |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.762742  |
|**`b2s`**  |**`bands`**  |**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.753472**|
|  `w2s`    |  `wise`     |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.772348  |
|  `w2s`    |  `wise`     |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.779303  |
|  `w2s`    |  `wise`     |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.753348  |
|  `w2s`    |  `wise`     |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.773345  |
|**`w2s`**  |**`wise`**   |**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.784288**|
|  `i2g`    |  `img`      |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.177365  |
|  `i2g`    |  `img`      |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  0.179617  |
|  `i2g`    |  `img`      |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.178693  |
|  `i2g`    |  `img`      |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  0.179261  |
|**`i2g`**  |**`img`**    |**`gz2c`**      |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.206522**|
|  `f2g`    |  `fits`     |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.278153  |
|  `f2g`    |  `fits`     |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  0.355011  |
|  `f2g`    |  `fits`     |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.273864  |
|  `f2g`    |  `fits`     |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  0.322159  |
|**`f2g`**  |**`fits`**   |**`gz2c`**      |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.372283**|
|  `s2g`    |  `spectra`  |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.268863  |
|  `s2g`    |  `spectra`  |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  0.184685  |
|  `s2g`    |  `spectra`  |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.267045  |
|  `s2g`    |  `spectra`  |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  0.192330  |
|**`s2g`**  |**`spectra`**|**`gz2c`**      |  **20**  |  **32**      |  **RMSProp** |  **categorical crossentropy**  |**0.225543**|
|  `ss2g`   |  `ssel`     |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.261824  |
|  `ss2g`   |  `ssel`     |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  0.182151  |
|  `ss2g`   |  `ssel`     |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.261932  |
|  `ss2g`   |  `ssel`     |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  0.256818  |
|**`ss2g`** |**`ssel`**   |**`gz2c`**      |  **20**  |  **64**      |  **RMSProp** |  **categorical crossentropy**  |**0.259511**|
|  `b2g`    |  `bands`    |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.246340  |
|  `b2g`    |  `bands`    |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  0.245495  |
|  `b2g`    |  `bands`    |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.245455  |
|  `b2g`    |  `bands`    |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  0.243466  |
|**`b2g`**  |**`bands`**  |**`gz2c`**      |  **20**  |  **32**      |  **RMSProp** |  **categorical crossentropy**  |**0.275815**|
|  `w2g`    |  `wise`     |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.257320  |
|  `w2g`    |  `wise`     |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  0.262387  |
|  `w2g`    |  `wise`     |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.258239  |
|  `w2g`    |  `wise`     |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  0.259943  |
|**`w2g`**  |**`wise`**   |**`gz2c`**      |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.254076**|

