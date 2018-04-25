import numpy as np
import gzip
import csv
from scipy.misc import imsave
import os
from glob import glob
import random
import sys
import tensorflow as tf
import threading
from tensorflow.examples.tutorials.mnist import input_data

# https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390
def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(28 * 28 * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, 28, 28, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


##################################################################################################

def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
  colorspace = 'grayscale'
  channels = 1
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example


class ImageCoder(object):
  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _is_png(filename):
  return '.png' in filename


def _process_image(filename, coder):
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    image_data = coder.png_to_jpeg(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards, output_directory):
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]

      try:
        image_buffer, height, width = _process_image(filename, coder)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % filename)
        continue

      example = _convert_to_example(filename, image_buffer, label,
                                    text, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

    writer.close()
    shard_counter = 0


def _process_image_files(name, filenames, texts, labels, num_shards, num_threads, output_directory):
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards, output_directory)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)

def _find_image_files(data_dir, labels_file):
    fpaths = glob(data_dir+"*.jpg")
    fpaths.sort()

    texts = []
    labels = []

    with open(labels_file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            l = row[1]
            texts.append(l)
            labels.append(int(l))

    shuffled_index = list(range(len(fpaths)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    fpaths = [fpaths[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    return fpaths, texts, labels


def _process_dataset(name, directory, num_shards, labels_file, num_threads, output_directory):
  filenames, texts, labels = _find_image_files(directory, labels_file)
  _process_image_files(name, filenames, texts, labels, num_shards, num_threads, output_directory)

#######################################################################################################################

def main():
    Dataset_dir = '/home/mikep/DataSets/MNIST/'

    # Download the dataset
    mnist = input_data.read_data_sets(Dataset_dir, one_hot=True, reshape=[])

    train_data_filename = Dataset_dir + 'train-images-idx3-ubyte.gz'
    train_labels_filename = Dataset_dir + 'train-labels-idx1-ubyte.gz'
    test_data_filename = Dataset_dir + 't10k-images-idx3-ubyte.gz'
    test_labels_filename = Dataset_dir + 't10k-labels-idx1-ubyte.gz'

    # Load the dataset into memory
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)

    img_save_dir = Dataset_dir + "images/"
    if not os.path.isdir(img_save_dir):
        os.mkdir(img_save_dir)

    labels_file = img_save_dir + "train-labels.csv"

    # Save the images to disk and generate a label file
    with open(labels_file, 'wb') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        for i in range(len(train_data)):
            imsave(img_save_dir + str(i) + ".jpg", train_data[i][:,:,0])
            writer.writerow([str(i) + ".jpg", train_labels[i]])

    # Generate TFRecord files of the data
    tfrecord_dir = Dataset_dir + "tfrecord/"
    if not os.path.isdir(tfrecord_dir):
        os.mkdir(tfrecord_dir)

    train_shards = 8
    num_threads = 8
    assert not train_shards % num_threads, ('Please make the num_threads commensurate with train_shards')

    _process_dataset("train", img_save_dir, train_shards, labels_file, num_threads, tfrecord_dir)

if __name__ == "__main__":
    main()
