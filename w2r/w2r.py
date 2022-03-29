
import tensorflow as tf

def model(norm):
    wise = tf.keras.Input(shape=(4), name='wise')
    x = norm(wise)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=wise, outputs=redshift, name='w2r')

    return model