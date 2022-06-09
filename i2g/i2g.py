
import tensorflow as tf
from astromlp.sdss.shared import CLASSES

def model():
    img = tf.keras.Input(shape=(150, 150, 3), name='img')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(img)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2')(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=img, outputs=gz2c, name='i2g')

    return model
