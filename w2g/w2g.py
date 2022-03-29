
import tensorflow as tf
from mysdss.shared import CLASSES

def model(norm):
    wise = tf.keras.Input(shape=(4), name='wise')
    x = norm(wise)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=wise, outputs=gz2c, name='w2g')

    return model