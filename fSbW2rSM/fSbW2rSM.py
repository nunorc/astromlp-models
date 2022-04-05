
import tensorflow as tf
from mysdss.shared import CLASSES

def model(snorm, bnorm, wnorm):
    # sub-network for fits
    fits = tf.keras.Input(shape=(61, 61, 5), name='fits')
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(fits)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.SeparableConv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    fits_branch = tf.keras.layers.Dense(32, activation='relu')(x)

    # sub-network for spectra
    spectra = tf.keras.Input(shape=(3522), name='spectra')
    x = snorm(spectra)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    spectra_branch = tf.keras.layers.Dense(32, activation='relu')(x)

    # sub-network for bands
    bands = tf.keras.Input(shape=(5), name='bands')
    x = bnorm(bands)
    bands_branch = tf.keras.layers.Dense(32, activation='relu')(x)

    # sub-network for wise
    wise = tf.keras.Input(shape=(4), name='wise')
    x = wnorm(wise)
    wise_branch = tf.keras.layers.Dense(32, activation='relu')(x)

    # concat inputs
    concat = tf.keras.layers.Concatenate()([fits_branch, spectra_branch, bands_branch, wise_branch])

    # top network
    x = tf.keras.layers.Dense(64, activation='relu')(concat)
    top = tf.keras.layers.Dense(32, activation='relu')(x)

    # output for redshift
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(top)

    # output for smass
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(top)

    model = tf.keras.Model(inputs=[fits, spectra, bands, wise], outputs=[redshift, smass], name='fSbW2rSM')

    return model
