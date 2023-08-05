import os
import tensorflow as tf
import time

from absl import app
from absl import flags
from preprocess import dataloader
from models.builder import process_model
from preprocess.utils import process_predicts, save_inference_time, get_model_path, get_model_path_test_results

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


def main(_):
        model_path = get_model_path()
        model_path_results = get_model_path_test_results()
        input_size = [FLAGS.image_size, FLAGS.image_size, 3]
        data = dataloader.DatasetProcessor(
                csv_file=FLAGS.test_filename,
                output_size=FLAGS.image_size,
                num_classes=FLAGS.num_classes,
                mode='test',
                seed=FLAGS.seed,
                is_training=False,
        )
        dataset, num_instances, num_classes = data.make_source_dataset()
        model = process_model(
                FLAGS.model_name, 
                input_size,
                FLAGS.num_classes,
                FLAGS.seed
        )
        model.summary()
        model.load_weights(model_path)
        y_pred = model.predict(dataset, verbose=1)
        data.only_label = True
        dataset, num_instances, num_classes = data.make_source_dataset()
        process_predicts(y_pred, dataset, model_path_results)


if __name__ == '__main__':
        with tf.device('/device:GPU:0'):
                app.run(main)
