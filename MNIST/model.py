import tensorflow as tf

# https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/blob/master/tensorflow_MNIST_DCGAN.py

#def lrelu(x, th=0.2):
#    return tf.maximum(th * x, x)

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        conv1 = tf.layers.batch_normalization(conv1, training=isTrain)
        conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
        #lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(conv1, 512, [4, 4], strides=(2, 2), padding='same')
        conv2 = tf.layers.batch_normalization(conv2, training=isTrain)
        conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
        #conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        #lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(conv2, 256, [4, 4], strides=(2, 2), padding='same')
        conv3 = tf.layers.batch_normalization(conv3, training=isTrain)
        conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
        #conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        #lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(conv3, 128, [4, 4], strides=(2, 2), padding='same')
        conv4 = tf.layers.batch_normalization(conv4, training=isTrain)
        conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)
        #conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        #lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(conv4, 1, [4, 4], strides=(2, 2), padding='same')
        #conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)

        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
        #lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(conv1, 256, [4, 4], strides=(2, 2), padding='same')
        conv2 = tf.layers.batch_normalization(conv2, training=isTrain)
        conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
        #conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        #lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(conv2, 512, [4, 4], strides=(2, 2), padding='same')
        conv3 = tf.layers.batch_normalization(conv3, training=isTrain)
        conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
        #conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        #lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(conv3, 1024, [4, 4], strides=(2, 2), padding='same')
        conv4 = tf.layers.batch_normalization(conv4, training=isTrain)
        conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)
        #conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        #lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(conv4, 1, [4, 4], strides=(1, 1), padding='valid')
        #conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv5)

        return o, conv5
