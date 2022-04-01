
import os, logging
import tensorflow as tf

logger = logging.getLogger(__name__)

model_store = './model_store'
models = ['i2r', 'f2r', 's2r', 'ss2r', 'b2r', 'w2r',
          'i2sm', 'f2sm', 's2sm', 'ss2sm', 'b2sm', 'w2sm',
          'i2s',  'f2s', 's2s', 'ss2s', 'b2s', 'w2s',
          'i2g', 'f2g', 's2g', 'ss2g', 'b2g', 'w2g']

h5_store = '/tmp/h5'
os.makedirs(h5_store, exist_ok=True)

for m in models:
	logger.info(f'Proc { m }')
	filename = os.path.join(model_store, m)
	model = tf.keras.models.load_model(filename)

	if os.path.exists(filename):
		dest = os.path.join(h5_store, f'{ m }.h5')
		model.save(dest, save_format='h5')
	else:
		logger.warn(f'{ file } not found.')
