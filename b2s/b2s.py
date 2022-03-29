
import tensorflow as tf
from mysdss.shared import CLASSES

def model(norm):
    bands = tf.keras.Input(shape=(5), name='bands')
    x = norm(bands)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=bands, outputs=subclass, name='b2s')

    return model