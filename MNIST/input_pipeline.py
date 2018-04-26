import tensorflow as tf

def parse_example_proto(example_serialized):
  # Dense features in Example proto.
  feature_map = {
      'image/height': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/width': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/channels': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  return features['image/encoded'], label, features['image/class/text']

def image_preprocessing(img_buffer):
    with tf.name_scope(name='decode_jpeg'):
        image = tf.image.decode_jpeg(img_buffer, channels=1)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        height = 64
        width = 64
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)
        #image = tf.image.resize_image_with_crop_or_pad(image, 28, 28)

        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        return image

def process(train_shards, batch_size=None, num_preprocess_threads=None, num_gpus=None):
    with tf.name_scope('batch_processing'):
        if train_shards is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        filename_queue = tf.train.string_input_producer(train_shards, shuffle=True, capacity=16)

        # Randomly deque images
        examples_per_shard = 7500
        min_queue_examples = examples_per_shard * num_gpus * batch_size
        examples_queue = tf.RandomShuffleQueue(capacity=min_queue_examples, min_after_dequeue=min_queue_examples/examples_per_shard, dtypes=[tf.string])

        # create queue of TFRecord file readers
        num_readers = 3
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()


        # Generate the images
        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, label_text = parse_example_proto(example_serialized)
            image = image_preprocessing(image_buffer)
            images_and_labels.append([image, label_index])


        # Form the batches
        images, label_index_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size, capacity=2 * num_preprocess_threads * batch_size)

        # Reshape images into these desired dimensions.
        height = 64
        width = 64
        depth = 1

        #images = tf.cast(images, tf.float32)
        #train_set = tf.image.resize_images(images, [height, width])
        #images = tf.reshape(images, shape=[batch_size, height, width, depth])
        labels = tf.reshape(label_index_batch, [batch_size])

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, labels

def input_pipeline(train_shards, batch_size=None, num_preprocess_threads=None, num_gpus=None):
    with tf.device('/cpu:0'):
        images, labels = process(train_shards, batch_size, num_preprocess_threads, num_gpus)

    return images, tf.random_normal([batch_size, 1, 1, 100]), labels
"""
def inputs(dataset, batch_size=None, num_preprocess_threads=None):
  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    images, labels = batch_inputs(
        dataset, batch_size, train=False,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=1)

  return images, labels

def distorted_inputs(dataset, batch_size=None, num_preprocess_threads=None):
  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.device('/cpu:0'):
    images, labels = batch_inputs(
        dataset, batch_size, train=True,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=FLAGS.num_readers)
  return images, labels

def decode_jpeg(image_buffer, scope=None):
  with tf.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def image_preprocessing(image_buffer, bbox, train, thread_id=0):
  if bbox is None:
    raise ValueError('Please supply a bounding box.')

  image = decode_jpeg(image_buffer)
  height = FLAGS.image_size
  width = FLAGS.image_size


  if train:
    image = distort_image(image, height, width, bbox, thread_id)
  else:
    resized = helper.IMAGE_SIZE
    image = tf.image.resize_image_with_crop_or_pad(image, resized, resized)

  image.set_shape([height,width,3])
  #image = tf.image.per_image_standardization(image)


  R, G, B = tf.split(value=image, num_or_size_splits=3, axis=2)
  R = tf.subtract(R, MEAN[0])
  R = tf.divide(R, STD[0])
  G = tf.subtract(G, MEAN[1])
  G = tf.divide(G, STD[1])
  B = tf.subtract(B, MEAN[2])
  B = tf.divide(B, STD[2])
  image = tf.stack([R,G,B],axis=2)


  image = tf.clip_by_value(image, 0.0, 1.0)
  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)


  return image


def parse_example_proto(example_serialized):
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None,
                 num_readers=1):
  with tf.name_scope('batch_processing'):
    data_files = dataset.data_files()
    if data_files is None:
      raise ValueError('No data files found for this dataset')

    # Create filename_queue
    filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=True,
                                                      capacity=16)

    # Approximate number of examples per shard.
    examples_per_shard = 7500
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    enqueue_ops = []
    for _ in range(num_readers):
        reader = dataset.reader()
        _, value = reader.read(filename_queue)
        enqueue_ops.append(examples_queue.enqueue([value]))

    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
    example_serialized = examples_queue.dequeue()

    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      image_buffer, label_index, bbox, _ = parse_example_proto(
          example_serialized)
      image = image_preprocessing(image_buffer, bbox, train, thread_id)
      images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size,
        capacity=2 * num_preprocess_threads * batch_size)

    # Reshape images into these desired dimensions.
    height = FLAGS.image_size
    width = FLAGS.image_size
    depth = 1

    #images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_index_batch, [batch_size])
"""
