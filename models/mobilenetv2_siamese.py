import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from absl import flags

FLAGS = flags.FLAGS


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )


def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def get_embedding_network(input_shape, seed):
  base_model = tf.keras.applications.MobileNetV2(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
  )
  if FLAGS.tag == '(tag:no_serengeti_weights)':
    base_model.trainable = True
  else:
    base_model.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
  x = keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))(x)
  return tf.keras.models.Model(
    inputs=[base_model.input],
    outputs=[x]
  )

def get_model(input_shape, num_classes, seed):
  input_1 = keras.layers.Input(input_shape)
  input_2 = keras.layers.Input(input_shape)
  embedding_network = get_embedding_network(input_shape, seed)
  tower_1 = embedding_network(input_1)
  tower_2 = embedding_network(input_2)
  
  merged = keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
  merged = keras.layers.BatchNormalization()(merged)
  x = keras.layers.Dense(num_classes, activation='sigmoid', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1245))(merged)

  model = keras.Model(inputs=[input_1, input_2], outputs=x)

  model.compile(
    loss=contrastive_loss,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy'])

  return model
