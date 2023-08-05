import pandas as pd
import tensorflow as tf

from absl import flags
from preprocess import preprocessing

AUTOTUNE = tf.data.experimental.AUTOTUNE

FLAGS = flags.FLAGS

class DatasetProcessor:

  def __init__(self,
      csv_file,
      is_training=False,
      output_size=224,
      resize_with_pad=False,
      num_classes=None,
      use_fake_data=False,
      mode='train',
      only_label=False,
      seed=None):
    self.csv_file = csv_file
    self.is_training = is_training
    self.output_size = output_size
    self.resize_with_pad = resize_with_pad
    self.num_classes = num_classes
    self.use_fake_data = use_fake_data
    self.only_label = only_label
    self.seed = seed
    self.mode = mode

  def get_dataframe(self):
    csv_data = pd.read_csv(self.csv_file)
    if self.num_classes is None:
      self.num_classes = len(csv_data.category.unique())
    print(csv_data.category.value_counts())
    return csv_data


  def load_dataset(self, csv_data):
    if FLAGS.model_type == 'siamese':
      file_name = (csv_data.file_name_x, csv_data.file_name_y)
    else:
      file_name = csv_data.file_name
    return  tf.data.Dataset.from_tensor_slices((file_name, csv_data.category))

  def _preprocess_image(self, image, label):
    if FLAGS.model_type == 'merge_mask':
      preprocess_image = tf.concat([preprocessing.preprocess_image(item,
                                    output_size=self.output_size,
                                    is_training=self.is_training,
                                    resize_with_pad=self.resize_with_pad) for item in image], axis=-1)

    else:
      preprocess_image = preprocessing.preprocess_image(image,
                                    output_size=self.output_size,
                                    is_training=self.is_training,
                                    is_siamese=FLAGS.model_type == 'siamese',
                                    resize_with_pad=self.resize_with_pad)
    return preprocess_image, label

  def make_source_dataset(self):
    csv_data = self.get_dataframe()
    num_instances = len(csv_data)

    dataset = self.load_dataset(csv_data)
    if self.is_training:
      dataset = dataset.shuffle(num_instances, seed=self.seed)
      dataset = dataset.repeat()

    def _load_image(file_name, label):
      label = tf.one_hot(label, self.num_classes)
      dir_images = FLAGS.images_path_ssd
      if FLAGS.model_type == 'siamese':
        image_x = tf.io.read_file(dir_images + file_name[0])
        image_y = tf.io.read_file(dir_images + file_name[1])
        image_x = tf.io.decode_jpeg(image_x, channels=3)
        image_y = tf.io.decode_jpeg(image_y, channels=3)
        return (image_x, image_y), label
      
      else:
        image = tf.io.read_file(dir_images + file_name)
        image = tf.io.decode_jpeg(image, channels=3)
        return image, label
        
    def _load_only_label(file_name, label):
        return file_name, tf.one_hot(label, self.num_classes)

    if self.only_label:
      dataset = dataset.map(_load_only_label, num_parallel_calls=AUTOTUNE)
    else:
      dataset = dataset.map(_load_image, num_parallel_calls=AUTOTUNE)
      dataset = dataset.map(self._preprocess_image, num_parallel_calls=AUTOTUNE)
          
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    if self.use_fake_data:
      dataset.take(1).repeat()

    return dataset, num_instances, self.num_classes
