import numpy as np
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf

def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def variable_with_weight_decay(name, shape, wd):
    dtype = tf.float32
    var = variable_on_cpu(
                        name,
                        shape,
                        tf.contrib.layers.variance_scaling_initializer())
    return var

def convolution(conv, shape, decay, stride, in_scope, is_training):
    with tf.variable_scope(in_scope) as scope:
        kernel = variable_with_weight_decay('weights',
                                            shape=shape,
                                            wd=decay)

        conv = tf.nn.conv2d(conv, kernel, [1, stride, stride, 1], padding='SAME')

    return conv

def convolution_transpose(conv, kernel_size, out_channels, stride, in_scope, padding='SAME'):
    with tf.variable_scope(in_scope) as scope:
        in_channels = conv.get_shape().as_list()[3]
        shape = [kernel_size, kernel_size, in_channels, out_channels]

        initializer = tf.contrib.layers.variance_scaling_initializer()
        kernel = variable_on_cpu('weights', shape, initializer)
        """
        kernel = variable_with_weight_decay('weights',
                                            shape=shape,
                                            wd=decay)
        """

        conv = tf.nn.conv2d(conv, kernel, [1, stride, stride, 1], padding=padding)

    return conv

fixed_z_ = np.random.normal(0, 1, (25, 1, 1, 100))
def show_result(sess, G_z, z, isTrain, num_epoch, show = False, save = False, path = 'result.png'):
#def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
