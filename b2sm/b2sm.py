
import tensorflow as tf

def model(norm):
    bands = tf.keras.Input(shape=(5), name='bands')
    x = norm(bands)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=bands, outputs=smass, name='b2sm')

    return model