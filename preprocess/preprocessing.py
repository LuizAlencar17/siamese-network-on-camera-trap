import random
import tensorflow as tf

from absl import flags

FLAGS = flags.FLAGS


def get_bbox_begin_and_bbox_size(image,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.65, 1],
                min_object_covered=0.5,
                max_attempts=100,
                seed=0):
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      area_range=area_range,
      aspect_ratio_range=aspect_ratio_range,
      use_image_if_no_bounding_boxes=True,
      max_attempts=max_attempts,
      seed=seed
  )
  return bbox_begin, bbox_size

def crop(image, bbox_begin, bbox_size):
  offset_height, offset_width, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  image = tf.image.crop_to_bounding_box(
    image,
    offset_height,
    offset_width,
    target_height,
    target_width
  )
  return image

def flip(image):
  return tf.image.flip_left_right(image)

def normalize_image(image):
  tf.compat.v1.logging.info('Normalizing inputs.')
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  mean = tf.constant([0.485, 0.456, 0.406])
  mean = tf.expand_dims(mean, axis=0)
  mean = tf.expand_dims(mean, axis=0)
  image = image - mean
  std = tf.constant([0.229, 0.224, 0.225])
  std = tf.expand_dims(std, axis=0)
  std = tf.expand_dims(std, axis=0)
  image = image/std
  return image

def scale_input_tf_mode(image):
  image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
  image = tf.cast(image, tf.float32)
  image /= 127.5
  image -= 1.
  return image

def scale_input(image):
  return tf.image.convert_image_dtype(image, dtype=tf.float32)
  # if FLAGS.input_scale_mode == 'torch_mode':
  #   return normalize_image(image)
  # elif FLAGS.input_scale_mode == 'tf_mode':
  #   return scale_input_tf_mode(image)
  # elif FLAGS.input_scale_mode == 'uint8':
  #   return tf.image.convert_image_dtype(image, dtype=tf.uint8)
  # else:
  #   return tf.image.convert_image_dtype(image, dtype=tf.float32)

def resize_image(image, output_size, resize_with_pad=False):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if resize_with_pad:
    image = tf.image.resize_with_pad(image, output_size, output_size)
  else:
    image = tf.image.resize(image, size=(output_size, output_size))
  return image

def data_augmentation_siamese(image_x, image_y, output_size):
    bbox_begin, bbox_size = get_bbox_begin_and_bbox_size(image_x)
    image_x = crop(image_x, bbox_begin, bbox_size)
    image_x = resize_image(image_x, output_size)
    image_y = crop(image_y, bbox_begin, bbox_size)
    image_y = resize_image(image_y, output_size)
    if random.randint(1, 2) % 2 == 0:
        image_x = flip(image_x)
        image_y = flip(image_y)
    return image_x, image_y

def data_augmentation(image, output_size, resize_with_pad):
    bbox_begin, bbox_size = get_bbox_begin_and_bbox_size(image)
    image = crop(image, bbox_begin, bbox_size)
    image = resize_image(image, output_size, resize_with_pad)
    if random.randint(1, 2) % 2 == 0:
        image = flip(image)
    return image

def preprocess_image(image,
                     output_size=224,
                     is_training=False,
                     resize_with_pad=False,
                     is_siamese=False):
  if is_siamese:
    image_x = resize_image(image[0], output_size, resize_with_pad)
    image_y = resize_image(image[1], output_size, resize_with_pad)
  else:
    image = resize_image(image, output_size, resize_with_pad)

  if is_training:
    if is_siamese:
      image_x, image_y = data_augmentation_siamese(image_x, image_y, output_size)
    else:
      image = data_augmentation(image, output_size, resize_with_pad)
      
  if is_siamese:
    return scale_input(image_x), scale_input(image_y)
  else:
    return scale_input(image)
