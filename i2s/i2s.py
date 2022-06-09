
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
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2')(x)
    subclass = tf.keras.layers.Dense(len(CLASSES['subclass']), activation='softmax', name='subclass')(x)

    model = tf.keras.Model(inputs=img, outputs=subclass, name='i2s')

    return model