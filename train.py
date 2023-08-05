import tensorflow as tf
import random
import os
import numpy as np

from absl import app
from absl import flags
from preprocess import dataloader
from models.builder import process_model
from preprocess.utils import save_history, get_model_path, get_history_name
from preprocess.utils import get_model_serengeti
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_classes', help='', default=1)
flags.DEFINE_integer('image_size', help='', default=256)
flags.DEFINE_integer('seed', help='', default=42)
flags.DEFINE_integer('num_epochs', help='', default=10)
flags.DEFINE_integer('patience', help='', default=10)
flags.DEFINE_integer('batch_size', help='', default=8)

flags.DEFINE_string('images_path', help='', default='')
flags.DEFINE_string('images_path_ssd', help='', default='')
flags.DEFINE_string('checkpoint_path', help='', default='')
flags.DEFINE_string('input_scale_mode', help='', default='')
flags.DEFINE_string('data_dir_mask', help='', default='')
flags.DEFINE_string('train_filename', help='', default='')
flags.DEFINE_string('val_filename', help='', default='')
flags.DEFINE_string('test_filename', help='', default='')

flags.DEFINE_string('dataset_name', help='', default='')
flags.DEFINE_string('model_name', help='', default='')
flags.DEFINE_string('model_type', help='', default='normal')
flags.DEFINE_string('tag', help='', default='serengeti_weights')

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def main(_):
  set_seeds(FLAGS.seed)
  input_shape = [FLAGS.image_size, FLAGS.image_size, 3]
  model_path = get_model_path()
  history_name = get_history_name()

  dataset, num_instances, num_classes = dataloader.DatasetProcessor(
    csv_file=FLAGS.train_filename,
    output_size=FLAGS.image_size,
    num_classes=FLAGS.num_classes,
    seed=FLAGS.seed,
    is_training=True,
  ).make_source_dataset()
  
  val_dataset, val_num_instances, num_classes = dataloader.DatasetProcessor(
    csv_file=FLAGS.val_filename,
    output_size=FLAGS.image_size,
    num_classes=FLAGS.num_classes,
    seed=FLAGS.seed,
    is_training=False,
  ).make_source_dataset()

  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    verbose=1,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
    )
  es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    verbose=1,
    patience=FLAGS.patience
    )
  
  model = process_model(FLAGS.model_name, input_shape, num_classes, FLAGS.seed)

  if FLAGS.tag != '(tag:no_serengeti_weights)':
    weights_path = get_model_serengeti()
    print('loading weights from:', weights_path)
    model.load_weights(weights_path)

  model.summary()
  history = model.fit(
    dataset,
    verbose=1,
    validation_data=val_dataset,
    callbacks=[cp_callback, es_callback],
    epochs=FLAGS.num_epochs,
    steps_per_epoch = num_instances // FLAGS.batch_size,
    validation_steps = val_num_instances // FLAGS.batch_size
    )
  save_history(history.history, history_name, FLAGS.num_epochs)

if __name__ == '__main__':
  with tf.device('/device:GPU:0'):
    app.run(main)
