
import tensorflow as tf

def model(norm):
    spectra = tf.keras.Input(shape=(3522), name='spectra')
    x = norm(spectra)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=spectra, outputs=smass, name='s2sm')

    return model