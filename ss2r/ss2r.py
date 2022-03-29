
import tensorflow as tf

def model(norm):
    ssel = tf.keras.Input(shape=(1423), name='ssel')
    x = norm(ssel)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    redshift = tf.keras.layers.Dense(1, activation='linear', name='redshift')(x)

    model = tf.keras.Model(inputs=ssel, outputs=redshift, name='ss2r')

    return model