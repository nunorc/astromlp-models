
import sys, os
import tensorflow as tf
import mlflow.keras

from astromlp.sdss.helper import Helper
from astromlp.sdss.utils import train_val_test_split, history_fit_plots, history_save, my_callbacks
from astromlp.sdss.datagen import DataGen

import f2sm

mlflow.tensorflow.autolog(log_models=False)
mlflow.set_tag('model', 'f2sm')

epochs = 20
batch_size = 32
loss = 'mae'
optimizer = 'adam'
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
ids = helper.ids_list(has_fits=True)
ids_train, ids_val, ids_test = train_val_test_split(ids)
train_gen = DataGen(ids_train, x=['fits'], y=['smass'], batch_size=batch_size, helper=helper)
val_gen = DataGen(ids_val, x=['fits'], y=['smass'], batch_size=batch_size, helper=helper)
test_gen = DataGen(ids_test, x=['fits'], y=['smass'], batch_size=batch_size, helper=helper)
mlflow.log_param('dataset_size', len(ids))

# model, compile and fit
model = f2sm.model()
model.compile(optimizer=opt, loss=loss, metrics=['mean_squared_error', 'mean_absolute_error'])
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=epochs, batch_size=batch_size,
                    callbacks=my_callbacks(), verbose=1)

# evaluate
score = model.evaluate(test_gen, batch_size=batch_size, return_dict=True)
mlflow.log_param('test_score', score)

# save model
model.save('../model_store/f2sm')

# save history and plots
history_save(model.name, history, base_dir='../model_history')
history_fit_plots(model.name, history, base_dir='../model_plots')
