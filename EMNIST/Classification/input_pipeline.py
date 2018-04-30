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
        #image = tf.image.resize_image_with_crop_or_pad(image, height, width)
        image = tf.image.resize_images(image, [height, width])

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
        images_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, label_text = parse_example_proto(example_serialized)
            image = image_preprocessing(image_buffer)
            images_labels.append([image, label_index])


        # Form the batches
        images, label_index_batch = tf.train.batch_join(images_labels, batch_size=batch_size, capacity=2 * num_preprocess_threads * batch_size)

        # Reshape images into these desired dimensions.
        height = 64
        width = 64
        depth = 1

        labels = tf.reshape(label_index_batch, [batch_size])

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, labels

def input_pipeline(train_shards, batch_size=None, num_preprocess_threads=None, num_gpus=None):
    with tf.device('/cpu:0'):
        images, labels = process(train_shards, batch_size, num_preprocess_threads, num_gpus)

    return images, labels, tf.random_normal([batch_size, 1, 1, 100])
