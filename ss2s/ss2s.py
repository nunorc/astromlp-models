
import tensorflow as tf
from astromlp.sdss.shared import CLASSES

def model(norm):
    ssel = tf.keras.Input(shape=(1423), name='ssel')
    x = norm(ssel)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=ssel, outputs=subclass, name='ss2s')

    return model