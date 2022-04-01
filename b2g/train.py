
import sys, os
import tensorflow as tf
import mlflow.keras

from mysdss.helper import Helper
from mysdss.utils import train_val_test_split
from mysdss.datagen import DataGen

import b2g

mlflow.tensorflow.autolog(log_models=False)
mlflow.set_tag('model', 'b2g')

epochs = 20
batch_size = 32
optimizer = 'rmsprop'
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
ids = helper.ids_list(has_bands=True, has_gz2c=True)
ids_train, ids_val, ids_test = train_val_test_split(ids)
train_gen = DataGen(ids_train, x=['bands'], y=['gz2c'], batch_size=batch_size, helper=helper)
val_gen = DataGen(ids_val, x=['bands'], y=['gz2c'], batch_size=batch_size, helper=helper)
test_gen = DataGen(ids_test, x=['bands'], y=['gz2c'], batch_size=batch_size, helper=helper)
mlflow.log_param('dataset_size', len(ids))

# normalization layer
norm = tf.keras.layers.Normalization()
tmp = helper.load_bands(ids)
norm.adapt(tmp)

# model, compile and fit
model = b2g.model(norm)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=epochs, batch_size=batch_size,
                    callbacks=my_callbacks(), verbose=1)

# evaluate
score = model.evaluate(test_gen, batch_size=batch_size, return_dict=True)

# save model
model.save(f'../model_store/{ model.name }')
