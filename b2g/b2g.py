
import tensorflow as tf
from mysdss.shared import CLASSES

def model(norm):
    bands = tf.keras.Input(shape=(5), name='bands')
    x = norm(bands)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=bands, outputs=gz2c, name='b2g')

    return model