
import tensorflow as tf
from astromlp.sdss.shared import CLASSES

def model(norm):
    spectra = tf.keras.Input(shape=(3522), name='spectra')
    x = norm(spectra)
    x = tf.keras.layers.Reshape((3522, 1), input_shape=(3522,))(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    gz2c = tf.keras.layers.Dense(len(CLASSES['gz2c']), activation='softmax', name='gz2c')(x)

    model = tf.keras.Model(inputs=spectra, outputs=gz2c, name='s2g')

    return model