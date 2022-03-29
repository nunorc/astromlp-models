
import tensorflow as tf

def model():
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
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=fits, outputs=smass, name='f2sm')

    return model