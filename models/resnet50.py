import tensorflow as tf

from absl import flags

FLAGS = flags.FLAGS

def get_model(input_shape, num_classes, seed):
  i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
  i = tf.cast(i, tf.float32)
  i = tf.keras.applications.resnet50.preprocess_input(i)
  base_model = tf.keras.applications.ResNet50(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
  )
  if FLAGS.tag == '(tag:no_serengeti_weights)':
    base_model.trainable = True
  else:
    base_model.trainable = False
  base_model = base_model(i)
  x = tf.keras.layers.GlobalAveragePooling2D()(base_model)
  x = tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))(x)
  x = tf.keras.layers.Dense(num_classes, activation="sigmoid", kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))(x)
  model = tf.keras.models.Model(
    inputs=[i],
    outputs=[x]
    )
  model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.SGD(),
      metrics=["accuracy"]
  )
  return model