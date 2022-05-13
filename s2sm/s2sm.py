
import tensorflow as tf

def model(norm):
    spectra = tf.keras.Input(shape=(3522), name='spectra')
    x = norm(spectra)
    x = tf.keras.layers.Reshape((3522, 1), input_shape=(3522,))(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    smass = tf.keras.layers.Dense(1, activation='linear', name='smass')(x)

    model = tf.keras.Model(inputs=spectra, outputs=smass, name='s2sm')

    return model
