
# astromlp-models

Collection of Deep Learning models to characterize different proprieties of galaxies,
based on data from the [Sloan Digital Sky Survey](https://www.sdss.org/) (SDSS).

The [SDSS Galaxy Subset](https://zenodo.org/record/6393488) dataset is used to
train and explore available models (the default location of the dataset w.r.t. to this repository
is `../sdss-gs`).

## Models

The following table quickly describes the available models.

|  Model   |  Input(s)    |  Output(s)     |  Type        |  Description      |
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
| `f2g`    |  `fits`      |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from FTIS data  |
| `s2g`    |  `spectra`   |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from spectra data  |
| `ss2g`   |  `ssel`      |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from selected spectra data  |
| `b2g`    |  `bands`     |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from bands data  |
| `w2sm`   |  `wise`      |  `gz2c`        |  classification  |  infer Galaxy Zoo 2 simplified class from WISE data  |

The models available in this repository are implemented using [Keras](https://keras.io/).
To fit the models available in this repository the [mysdss](https://github.com/nunorc/mysdss)
Python companion package is also required.

You can fit a model using [mlflow](https://mlflow.org/), for example to fit the `i2r` model using
your current `python` (i.e. don't create a new environment using `conda`) you can run:

    $ mlflow run i2r --no-conda

You can also change the parameters to run the model, namely the number of epochs, the batch size,
the loss function and optimizer to use, for example:

    $ mlflow run i2r -P epochs=10 -P batch_size=32 -P loss=mse -P optimizer=adam --no-conda

You can also change the location of the dataset by setting the `ds` parameter:

    $ mlflow run i2r -P ds=/tmp/sdss-gs --no-conda

To view the data concerting the fitting of the available models you can use `mlflow` user interface:

    $ mlflow ui

And can also check the generated [tensorboard](https://www.tensorflow.org/tensorboard) logs, for
example:

    $ tensorboard --logdir i2r/logs/

## Inputs & Outputs

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
|  `smass`     |  regression   |  stellar mass extracted from the [eBOSS Firefly catalog]("https://www.sdss.org/dr16/spectro/eboss-firefly-value-added-catalog)  |
|  `gz2c`      |  classification  | simplified version of the Galaxy Zoo 2 classification, from [Willett et al 2013]("https://academic.oup.com/mnras/article/435/4/2835/1022913) (see class sets section below for details)  |

## Class Sets

The following tables describe the class set labels for the classification outputs.

### SDSS sub-class subset (`subclass`)

The sub-class parameter for each object is available from the
[SDSS spectroscopic catalogs](https://www.sdss.org/dr17/spectro/catalogs/).

|  Label        |  Description  |
|:-------------:|---------------|
|  `AGN`        |  has detectable emission lines that are consistent with being a Seyfert or LINER  |
|  `BROADLINE`  |  has lines detected at the 10-sigma level with sigmas > 200 km/sec at the 5-sigma level  |
|  `STARBURST`  |  galaxy is star-forming  |
|  `STARFORMING`|  has detectable emission lines that are consistent with star-formation criteria  |

### Galaxy Zoo 2 Simplified Classes (`gz2c`)

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


## Single Input/Output Models Hyper-parameters Exploration

The following table summarizes different combinations of batch size (32 or 64),
optimizer (RMSProp or Adam) and loss functions (Mean Squared Error or Mean Absolute
Error for regression, categorical crossentropy is always used for multi-label
classification) hyper-parameters exploration, the score is the best validation value for the
validation dataset. The highlighted row describes the final hyper-parameters used for
bootstrapping each model and the corresponding evaluation score (accuracy for classification
models) on the test set, never seen by any model during the exploration steps.

|  Input      |  Output        |  Epochs  |  Batch Size  |  Optimizer   |  Loss     |   Score    |
|:-----------:|:--------------:|:--------:|:------------:|:------------:|:---------:|:----------:|
|  `img`      |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.003365  |
|  `img`      |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.030170  |
|  `img`      |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.016661  |
|  `img`      |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.071612  |
|  `img`      |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.003508  |
|  `img`      |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.030480  |
|  `img`      |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.003498  |
|  `img`      |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.071558  |
|**`img`**    |**`redshift`**  |  **20**  |  **32**      |  **RMSProp** |  **MSE**  |**0.002377**|
|  `fits`     |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.001930  |
|  `fits`     |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.027434  |
|  `fits`     |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.001917  |
|  `fits`     |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.024896  |
|  `fits`     |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.002075  |
|  `fits`     |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.028206  |
|  `fits`     |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.001870  |
|  `fits`     |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.024997  |
|**`fits`**   |**`redshift`**  |  **20**  |  **64**      |  **Adam**    |  **MSE**  |**0.002020**|
|  `spectra`  |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.004649  |
|  `spectra`  |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.033334  |
|  `spectra`  |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.006617  |
|  `spectra`  |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.040978  |
|  `spectra`  |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.005554  |
|  `spectra`  |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.038496  |
|  `spectra`  |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.014275  |
|  `spectra`  |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.044180  |
|**`spectra`**|**`redshift`**  |  **20**  |  **32**      |  **RMSProp** |  **MSE**  |**0.004401**|
|  `ssel`     |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.004343  |
|  `ssel`     |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.030403  |
|  `ssel`     |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.008098  |
|  `ssel`     |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.045030  |
|  `ssel`     |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.005174  |
|  `ssel`     |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.031783  |
|  `ssel`     |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.006342  |
|  `ssel`     |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.053812  |
|**`ssel`**   |**`redshift`**  |  **20**  |  **32**      |  **RMSProp** |  **MSE**  |**0.003700**|
|  `bands`    |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.003765  |
|  `bands`    |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.027912  |
|  `bands`    |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.004437  |
|  `bands`    |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.027528  |
|  `bands`    |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.004726  |
|  `bands`    |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.029070  |
|  `bands`    |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.004593  |
|  `bands`    |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.027921  |
|**`bands`**  |**`redshift`**  |  **20**  |  **32**      |  **RMSProp** |  **MSE**  |**0.002806**|
|  `wise`     |  `redshift`    |  10      |  32          |  RMSProp     |  MSE      |  0.011475  |
|  `wise`     |  `redshift`    |  10      |  32          |  RMSProp     |  MAE      |  0.055713  |
|  `wise`     |  `redshift`    |  10      |  32          |  Adam        |  MSE      |  0.011523  |
|  `wise`     |  `redshift`    |  10      |  32          |  Adam        |  MAE      |  0.055016  |
|  `wise`     |  `redshift`    |  10      |  64          |  RMSProp     |  MSE      |  0.011657  |
|  `wise`     |  `redshift`    |  10      |  64          |  RMSProp     |  MAE      |  0.056263  |
|  `wise`     |  `redshift`    |  10      |  64          |  Adam        |  MSE      |  0.011557  |
|  `wise`     |  `redshift`    |  10      |  64          |  Adam        |  MAE      |  0.055382  |
|**`wise`**   |**`redshift`**  | **20**   |  **32**      |  **RMSProp** |  **MSE**  |**0.012004**|
|  `img`      |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  40929.859 |
|  `img`      |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  26.048    |
|  `img`      |  `smass`       |  10      |  32          |  Adam        |  MSE      |  40813.363 |
|  `img`      |  `smass`       |  10      |  32          |  Adam        |  MAE      |  26.695    |
|  `img`      |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  40982.949 |
|  `img`      |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  26.736    |
|  `img`      |  `smass`       |  10      |  64          |  Adam        |  MSE      |  42835.566 |
|  `img`      |  `smass`       |  10      |  64          |  Adam        |  MAE      |  26.238    |
|**`img`**    |**`smass`**     |  **20**  |  **64**      |  **Adam**    |  **MAE**  |**22.537**  |
|  `fits`     |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  40600.176 |
|  `fits`     |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  27.171    |
|  `fits`     |  `smass`       |  10      |  32          |  Adam        |  MSE      |  40491.633 |
|  `fits`     |  `smass`       |  10      |  32          |  Adam        |  MAE      |  25.758    |
|  `fits`     |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  40679.352 |
|  `fits`     |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  26.600    |
|  `fits`     |  `smass`       |  10      |  64          |  Adam        |  MSE      |  40903.289 |
|  `fits`     |  `smass`       |  10      |  64          |  Adam        |  MAE      |  25.931    |
|**`fits`**   |**`smass`**     |  **20**  |  **32**      |  **Adam**    |  **MAE**  |**21.753**  |
|  `spectra`  |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  2963.335  |
|  `spectra`  |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  20.871    |
|  `spectra`  |  `smass`       |  10      |  32          |  Adam        |  MSE      |  3432.733  |
|  `spectra`  |  `smass`       |  10      |  32          |  Adam        |  MAE      |  20.129    |
|  `spectra`  |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  3159.474  |
|  `spectra`  |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  21.317    |
|  `spectra`  |  `smass`       |  10      |  64          |  Adam        |  MSE      |  3518.646  |
|  `spectra`  |  `smass`       |  10      |  64          |  Adam        |  MAE      |  20.420    |
|**`spectra`**|**`smass`**     |  **20**  |  **32**      |  **Adam**    |  **MAE**  |**19.577**  |
|  `ssel`     |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  40206.445 |
|  `ssel`     |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  22.797    |
|  `ssel`     |  `smass`       |  10      |  32          |  Adam        |  MSE      |  3018.159  |
|  `ssel`     |  `smass`       |  10      |  32          |  Adam        |  MAE      |  22.728    |
|  `ssel`     |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  40271.656 |
|  `ssel`     |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  24.099    |
|  `ssel`     |  `smass`       |  10      |  64          |  Adam        |  MSE      |  40798.273 |
|  `ssel`     |  `smass`       |  10      |  64          |  Adam        |  MAE      |  22.380    |
|**`ssel`**   |**`smass`**     |  **20**  |  **64**      |  **Adam**    |  **MAE**  |**18.793**  |
|  `bands`    |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  3387.860  |
|  `bands`    |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  26.523    |
|  `bands`    |  `smass`       |  10      |  32          |  Adam        |  MSE      |  5747.023  |
|  `bands`    |  `smass`       |  10      |  32          |  Adam        |  MAE      |  25.930    |
|  `bands`    |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  5520.346  |
|  `bands`    |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  26.694    |
|  `bands`    |  `smass`       |  10      |  64          |  Adam        |  MSE      |  5725.097  |
|  `bands`    |  `smass`       |  10      |  64          |  Adam        |  MAE      |  25.151    |
|**`bands`**  |**`smass`**     |  **20**  |  **64**      |  **Adam**    |  **MAE**  |**25.344**  |
|  `wise`     |  `smass`       |  10      |  32          |  RMSProp     |  MSE      |  4323.260  |
|  `wise`     |  `smass`       |  10      |  32          |  RMSProp     |  MAE      |  30.034    |
|  `wise`     |  `smass`       |  10      |  32          |  Adam        |  MSE      |  4391.042  |
|  `wise`     |  `smass`       |  10      |  32          |  Adam        |  MAE      |  29.858    |
|  `wise`     |  `smass`       |  10      |  64          |  RMSProp     |  MSE      |  4423.225  |
|  `wise`     |  `smass`       |  10      |  64          |  RMSProp     |  MAE      |  30.605    |
|  `wise`     |  `smass`       |  10      |  64          |  Adam        |  MSE      |  4433.951  |
|  `wise`     |  `smass`       |  10      |  64          |  Adam        |  MAE      |  29.800    |
|**`wise`**   |**`smass`**     |  **20**  |  **64**      |  **Adam**    |  **MAE**  |**27.719**  |
|  `img`      |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.771606  |
|  `img`      |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.778097  |
|  `img`      |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.767020  |
|  `img`      |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.787760  |
|**`img`**    |**`subclass`**  |  **20**  |  **64**      |  **Adam**    |  **categorical crossentropy**  |**0.765625**|
|  `fits`     |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.771977  |
|  `fits`     |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.782177  |
|  `fits`     |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.766183  |
|  `fits`     |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.778739  |
|**`fits`**   |**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.785590**|
|  `spectra`  |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.765579  |
|  `spectra`  |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.767897  |
|  `spectra`  |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.766648  |
|  `spectra`  |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.713914  |
|**`spectra`**|**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.763455**|
|  `ssel`     |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.765764  |
|  `ssel`     |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.768824  |
|  `ssel`     |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.761719  |
|  `ssel`     |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.768415  |
|**`ssel`**   |**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.756076**|
|  `bands`    |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.758995  |
|  `bands`    |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.762148  |
|  `bands`    |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.757720  |
|  `bands`    |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.762742  |
|**`bands`**  |**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.753472**|
|  `wise`     |  `subclass`    |  10      |  32          |  RMSProp     |  categorical crossentropy      |  0.772348  |
|  `wise`     |  `subclass`    |  10      |  32          |  Adam        |  categorical crossentropy      |  0.779303  |
|  `wise`     |  `subclass`    |  10      |  64          |  RMSProp     |  categorical crossentropy      |  0.753348  |
|  `wise`     |  `subclass`    |  10      |  64          |  Adam        |  categorical crossentropy      |  0.773345  |
|**`wise`**   |**`subclass`**  |  **20**  |  **32**      |  **Adam**    |  **categorical crossentropy**  |**0.784288**|
|  `img`      |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `img`      |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  FIXME     |
|  `img`      |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `img`      |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  FIXME     |
|**`img`**    |**`gz2c`**      |  **20**  |  **64**      |  **Adam**    |  **categorical crossentropy**  |**FIXME**   |
|  `fits`     |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `fits`     |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  FIXME     |
|  `fits`     |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `fits`     |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  FIXME     |
|**`fits`**   |**`gz2c`**      |  **20**  |  **64**      |  **Adam**    |  **categorical crossentropy**  |**FIXME**   |
|  `spectra`  |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `spectra`  |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  FIXME     |
|  `spectra`  |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `spectra`  |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  FIXME     |
|**`spectra`**|**`gz2c`**      |  **20**  |  **64**      |  **Adam**    |  **categorical crossentropy**  |**FIXME**   |
|  `ssel`     |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `ssel`     |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  FIXME     |
|  `ssel`     |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `ssel`     |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  FIXME     |
|**`ssel`**   |**`gz2c`**      |  **20**  |  **64**      |  **Adam**    |  **categorical crossentropy**  |**FIXME**   |
|  `bands`    |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `bands`    |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  FIXME     |
|  `bands`    |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `bands`    |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  FIXME     |
|**`bands`**  |**`gz2c`**      |  **20**  |  **64**      |  **Adam**    |  **categorical crossentropy**  |**FIXME**   |
|  `wise`     |  `gz2c`        |  10      |  32          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `wise`     |  `gz2c`        |  10      |  32          |  Adam        |  categorical crossentropy      |  FIXME     |
|  `wise`     |  `gz2c`        |  10      |  64          |  RMSProp     |  categorical crossentropy      |  FIXME     |
|  `wise`     |  `gz2c`        |  10      |  64          |  Adam        |  categorical crossentropy      |  FIXME     |
|**`wise`**   |**`gz2c`**      |  **20**  |  **64**      |  **Adam**    |  **categorical crossentropy**  |**FIXME**   |

