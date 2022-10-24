
import sys, os
import tensorflow as tf
import mlflow.keras

from astromlp.sdss.helper import Helper
from astromlp.sdss.utils import train_val_test_split, history_fit_plots, history_save, my_callbacks
from astromlp.sdss.datagen import DataGen

import fSbW2sG

mlflow.tensorflow.autolog(log_models=False)
mlflow.set_tag('model', 'fSbW2sG')

epochs = 30
batch_size = 32
optimizer = 'adam'
ds = '../../sdss-gs'

if len(sys.argv)>1:
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    optimizer = sys.argv[3]
    ds = sys.argv[4]

# data helper for sdss-gs
helper = Helper(ds=ds)

# optimizers
opt = tf.keras.optimizers.Adam()
if optimizer == 'rmsprop':
    opt = tf.keras.optimizers.RMSprop()

# ids and datagens
ids = helper.ids_list(has_fits=True, has_spectra=True, has_bands=True, has_wise=True, has_gz2c=True)
ids_train, ids_val, ids_test = train_val_test_split(ids)
train_gen = DataGen(ids_train, x=['fits','spectra','bands','wise'], y=['subclass','gz2c'], batch_size=batch_size, helper=helper)
val_gen = DataGen(ids_val, x=['fits','spectra','bands','wise'], y=['subclass','gz2c'], batch_size=batch_size, helper=helper)
test_gen = DataGen(ids_test, x=['fits','spectra','bands','wise'], y=['subclass','gz2c'], batch_size=batch_size, helper=helper)
mlflow.log_param('dataset_size', len(ids))

# normalizers
snorm = tf.keras.layers.Normalization()
tmp = helper.load_spectras(ids)
snorm.adapt(tmp)
bnorm = tf.keras.layers.Normalization()
tmp = helper.load_bands(ids)
bnorm.adapt(tmp)
wnorm = tf.keras.layers.Normalization()
tmp = helper.load_wises(ids)
wnorm.adapt(tmp)
del tmp

# model, compile and fit
model = fSbW2sG.model(snorm, bnorm, wnorm)
loss_funs = {
    'subclass': 'categorical_crossentropy',
    'gz2c': 'categorical_crossentropy'
}
metrics = {
    'subclass': 'accuracy',
    'gz2c': 'accuracy'
}
model.compile(optimizer=opt, loss=loss_funs, metrics=metrics)
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=epochs, batch_size=batch_size,
                    callbacks=my_callbacks(), verbose=1)

# evaluate
score = model.evaluate(test_gen, batch_size=batch_size, return_dict=True)
mlflow.log_param('test_score', score)

# save model
model.save(f'../model_store/{ model.name }')

# save history and plots
history_save(model.name, history, base_dir='../model_history')
history_fit_plots(model.name, history, base_dir='../model_plots')
