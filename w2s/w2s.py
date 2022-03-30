
import tensorflow as tf
from mysdss.shared import CLASSES

def model(norm):
    wise = tf.keras.Input(shape=(4), name='wise')
    x = norm(wise)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=wise, outputs=subclass, name='w2s')

    return model