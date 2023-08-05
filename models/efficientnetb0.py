import tensorflow as tf

from absl import flags

FLAGS = flags.FLAGS

def get_model(input_shape, num_classes, seed):
  base_model = tf.keras.applications.EfficientNetB0(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
  )
  if FLAGS.tag == '(tag:no_serengeti_weights)':
    base_model.trainable = True
  else:
    base_model.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
  x = tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))(x)
  x = tf.keras.layers.Dense(num_classes, activation="sigmoid", kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))(x)
  model = tf.keras.models.Model(
    inputs=[base_model.input],
    outputs=[x]
    )
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.SGD(),
      metrics=["accuracy"]
  )
  return model