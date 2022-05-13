
import sys, os
import tensorflow as tf
import mlflow.keras

from astromlp.sdss.helper import Helper
from astromlp.sdss.utils import train_val_test_split, history_fit_plots, my_callbacks
from astromlp.sdss.datagen import DataGen

import s2r

mlflow.tensorflow.autolog(log_models=False)
mlflow.set_tag('model', 's2r')

epochs = 100
batch_size = 32
loss = 'mse'
optimizer = 'rmsprop'
ds = '../../sdss-gs'

if len(sys.argv)>1:
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    loss = sys.argv[3]
    optimizer = sys.argv[4]
    ds = sys.argv[5]

# data helper for sdss-gs
helper = Helper(ds=ds)

# optimizers
opt = tf.keras.optimizers.Adam()
if optimizer == 'rmsprop':
    opt = tf.keras.optimizers.RMSprop()

# ids and datagens
ids = helper.ids_list(has_spectra=True)
ids_train, ids_val, ids_test = train_val_test_split(ids)
train_gen = DataGen(ids_train, x=['spectra'], y=['redshift'], batch_size=batch_size, helper=helper)
val_gen = DataGen(ids_val, x=['spectra'], y=['redshift'], batch_size=batch_size, helper=helper)
test_gen = DataGen(ids_test, x=['spectra'], y=['redshift'], batch_size=batch_size, helper=helper)
mlflow.log_param('dataset_size', len(ids))

# normalization layer
norm = tf.keras.layers.Normalization()
tmp = helper.load_spectras(ids)
norm.adapt(tmp)

# model, compile and fit
model = s2r.model(norm)
model.compile(optimizer=opt, loss=loss, metrics=['mean_squared_error', 'mean_absolute_error'])
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=epochs, batch_size=batch_size,
                    callbacks=my_callbacks(lr_scheduler=True, schedule='step_decay'), verbose=1)

# evaluate
score = model.evaluate(test_gen, batch_size=batch_size, return_dict=True)

# save model
model.save('../model_store/s2r')

# save history plots
history_fit_plots(model.name, history, base_dir='../model_plots')

