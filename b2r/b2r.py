
import tensorflow as tf

def model(norm):
    bands = tf.keras.Input(shape=(5), name='bands')
    x = norm(bands)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=bands, outputs=redshift, name='b2r')

    return model