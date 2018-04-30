import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def compound_atrous_conv(input, output, shape, stride, rate, name, is_training=True):
    with slim.arg_scope([slim.conv2d],
                         activation_fn=None,
                         padding='VALID',
                         biases_initializer=None):

        conv = slim.conv2d(inputs=input, num_outputs=output, kernel_size=shape, stride=stride, rate=rate, scope=name, trainable=False)
        conv = tf.layers.batch_normalization(conv, momentum=.95, epsilon=1e-5, fused=True, training=is_training, name=name+'_bn', trainable=False)

        conv = tf.nn.relu(conv, name=name+'_bn_relu')

        return conv

def compound_conv(input, output, shape, stride, name, relu=True, padding='VALID', trainable=False, is_training=True):
    with slim.arg_scope([slim.conv2d],
                         activation_fn=None,
                         padding=padding,
                         biases_initializer=None):

        conv = slim.conv2d(inputs=input, num_outputs=output, kernel_size=shape, stride=stride, scope=name, trainable=trainable)
        conv = tf.layers.batch_normalization(conv, momentum=.95, epsilon=1e-5, fused=True, training=is_training, name=name+'_bn', trainable=trainable)


        if relu == True:
            conv = tf.nn.relu(conv, name=name+'_bn_relu')

        return conv

def skip_connection(in1, in2, name):
    add = tf.add_n([in1, in2], name=name)
    add = tf.nn.relu(add, name=name+'_relu')

    return add

def block(input, outs, sizes, strides, pad, names, rate=None, trainable=False):
    conv1 = compound_conv(input, outs[0], sizes[0], strides[0], names[0], trainable=trainable)

    pad =  tf.pad(conv1, paddings=np.array([[0,0], [pad, pad], [pad, pad], [0, 0]]), name=names[1])

    if rate != None:
        conv2 = compound_atrous_conv(pad, outs[1], sizes[1], strides[1], rate, names[2])
    else:
        conv2 = compound_conv(pad, outs[1], sizes[1], strides[1], names[2], trainable=trainable)

    conv3 = compound_conv(conv2, outs[2], sizes[2], strides[2], names[3], relu=False, trainable=trainable)

    return conv3

def ResNet101(input):
    conv1_1_3x3_s2 = compound_conv(input, 64, 3, 2, 'conv1_1_3x3_s2', padding='SAME')

    conv1_2_3x3 = compound_conv(conv1_1_3x3_s2, 64, 3, 1, 'conv1_2_3x3', padding='SAME')

    conv1_3_3x3 = compound_conv(conv1_2_3x3, 128, 3, 1, 'conv1_3_3x3', padding='SAME')

    pool1_3x3_s2 = tf.nn.max_pool(conv1_3_3x3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1_3x3_s2')

    conv2_1_1x1_proj = compound_conv(pool1_3x3_s2, 256, 1, 1, 'conv2_1_1x1_proj', relu=False)

    ###################################

    outs = [64, 64, 256]
    sizes = [1, 3, 1]
    strides = [1, 1, 1]
    pad = 1
    names = ['conv2_1_1x1_reduce', 'padding1', 'conv2_1_3x3', 'conv2_1_1x1_increase']
    conv2_1_1x1_increase = block(pool1_3x3_s2, outs, sizes, strides, pad, names)

    #####################################

    conv2_1 = skip_connection(conv2_1_1x1_proj, conv2_1_1x1_increase, 'conv2_1')

    outs = [64, 64, 256]
    sizes = [1, 3, 1]
    strides = [1, 1, 1]
    pad = 1
    names = ['conv2_2_1x1_reduce', 'padding2', 'conv2_2_3x3', 'conv2_2_1x1_increase']
    conv2_2_1x1_increase = block(conv2_1, outs, sizes, strides, pad, names)

    ####################################

    conv2_2 = skip_connection(conv2_1, conv2_2_1x1_increase, 'conv2_2')

    outs = [64, 64, 256]
    sizes = [1, 3, 1]
    strides = [1, 1, 1]
    pad = 1
    names = ['conv2_3_1x1_reduce', 'padding3', 'conv2_3_3x3', 'conv2_3_1x1_increase']
    conv2_3_1x1_increase = block(conv2_2, outs, sizes, strides, pad, names)

    ########################################

    conv2_3 = skip_connection(conv2_2, conv2_3_1x1_increase, 'conv2_3')

    conv3_1_1x1_proj = compound_conv(conv2_3, 512, 1, 2, 'conv3_1_1x1_proj', relu=False)

    ########################################

    outs = [128, 128, 512]
    sizes = [1, 3, 1]
    strides = [2, 1, 1]
    pad = 1
    names = ['conv3_1_1x1_reduce', 'padding4', 'conv3_1_3x3', 'conv3_1_1x1_increase']
    conv3_1_1x1_increase = block(conv2_3, outs, sizes, strides, pad, names)

    ###########################################

    conv3_1 = skip_connection(conv3_1_1x1_proj, conv3_1_1x1_increase, 'conv3_1')

    outs = [128, 128, 512]
    sizes = [1, 3, 1]
    strides = [1, 1, 1]
    pad = 1
    names = ['conv3_2_1x1_reduce', 'padding5', 'conv3_2_3x3', 'conv3_2_1x1_increase']
    conv3_2_1x1_increase = block(conv3_1, outs, sizes, strides, pad, names)

    ##############################################

    conv3_2 = skip_connection(conv3_1, conv3_2_1x1_increase, 'conv3_2')

    outs = [128, 128, 512]
    sizes = [1, 3, 1]
    strides = [1, 1, 1]
    pad = 1
    names = ['conv3_3_1x1_reduce', 'padding6', 'conv3_3_3x3', 'conv3_3_1x1_increase']
    conv3_3_1x1_increase = block(conv3_2, outs, sizes, strides, pad, names)

    #############################################

    conv3_3 = skip_connection(conv3_2, conv3_3_1x1_increase, 'conv3_3')

    outs = [128, 128, 512]
    sizes = [1, 3, 1]
    strides = [1, 1, 1]
    pad = 1
    names = ['conv3_4_1x1_reduce', 'padding7', 'conv3_4_3x3', 'conv3_4_1x1_increase']
    conv3_4_1x1_increase = block(conv3_3, outs, sizes, strides, pad, names)

    ##############################################

    conv3_4 = skip_connection(conv3_3, conv3_4_1x1_increase, 'conv3_4')

    conv4_1_1x1_proj = compound_conv(conv3_4, 1024, 1, 1, 'conv4_1_1x1_proj', relu=False)

    ###############################################

    outs = [256, 256, 1024]
    sizes = [1, 3, 1]
    strides = [1, 1, 1]
    pad = 2
    names = ['conv4_1_1x1_reduce', 'padding8', 'conv4_1_3x3', 'conv4_1_1x1_increase']
    conv4_1_1x1_increase = block(conv3_4, outs, sizes, strides, pad, names, rate=2)

    ##################################################

    conv4_names = [['conv4_2_1x1_reduce', 'padding9', 'conv4_2_3x3', 'conv4_2_1x1_increase'],
                   ['conv4_3_1x1_reduce', 'padding10', 'conv4_3_3x3', 'conv4_3_1x1_increase'],
                   ['conv4_4_1x1_reduce', 'padding11', 'conv4_4_3x3', 'conv4_4_1x1_increase'],
                   ['conv4_5_1x1_reduce', 'padding12', 'conv4_5_3x3', 'conv4_5_1x1_increase'],
                   ['conv4_6_1x1_reduce', 'padding13', 'conv4_6_3x3', 'conv4_6_1x1_increase'],
                   ['conv4_7_1x1_reduce', 'padding14', 'conv4_7_3x3', 'conv4_7_1x1_increase'],
                   ['conv4_8_1x1_reduce', 'padding15', 'conv4_8_3x3', 'conv4_8_1x1_increase'],
                   ['conv4_9_1x1_reduce', 'padding16', 'conv4_9_3x3', 'conv4_9_1x1_increase'],
                   ['conv4_10_1x1_reduce', 'padding17', 'conv4_10_3x3', 'conv4_10_1x1_increase'],
                   ['conv4_11_1x1_reduce', 'padding18', 'conv4_11_3x3', 'conv4_11_1x1_increase'],
                   ['conv4_12_1x1_reduce', 'padding19', 'conv4_12_3x3', 'conv4_12_1x1_increase'],
                   ['conv4_13_1x1_reduce', 'padding20', 'conv4_13_3x3', 'conv4_13_1x1_increase'],
                   ['conv4_14_1x1_reduce', 'padding21', 'conv4_14_3x3', 'conv4_14_1x1_increase'],
                   ['conv4_15_1x1_reduce', 'padding22', 'conv4_15_3x3', 'conv4_15_1x1_increase'],
                   ['conv4_16_1x1_reduce', 'padding23', 'conv4_16_3x3', 'conv4_16_1x1_increase'],
                   ['conv4_17_1x1_reduce', 'padding24', 'conv4_17_3x3', 'conv4_17_1x1_increase'],
                   ['conv4_18_1x1_reduce', 'padding25', 'conv4_18_3x3', 'conv4_18_1x1_increase'],
                   ['conv4_19_1x1_reduce', 'padding26', 'conv4_19_3x3', 'conv4_19_1x1_increase'],
                   ['conv4_20_1x1_reduce', 'padding27', 'conv4_20_3x3', 'conv4_20_1x1_increase'],
                   ['conv4_21_1x1_reduce', 'padding28', 'conv4_21_3x3', 'conv4_21_1x1_increase'],
                   ['conv4_22_1x1_reduce', 'padding29', 'conv4_22_3x3', 'conv4_22_1x1_increase'],
                   ['conv4_23_1x1_reduce', 'padding30', 'conv4_23_3x3', 'conv4_23_1x1_increase']]

    outs = [256, 256, 1024]
    sizes = [1, 3, 1]
    strides = [1, 1, 1]
    pad = 2

    conv4_i = conv4_1_1x1_proj
    conv4_i_1x1_increase = conv4_1_1x1_increase

    conv4_i_outputs = []
    conv4_i_1x1_increase_outputs = []
    for name, i in zip(conv4_names, range(len(conv4_names))):
        i += 1
        conv4_i = skip_connection(conv4_i, conv4_i_1x1_increase, 'conv4_'+str(i))
        conv4_i_outputs.append(conv4_i)

        conv4_i_1x1_increase = block(conv4_i, outs, sizes, strides, pad, name, rate=2)
        conv4_i_1x1_increase_outputs.append(conv4_i_1x1_increase)

    ##############################################################

    conv4_22 = conv4_i_outputs[-1]
    conv4_23_1x1_increase = conv4_i_1x1_increase_outputs[-1]

    ###################################################################

    conv4_23 = skip_connection(conv4_22, conv4_23_1x1_increase, 'conv4_23')

    k = conv4_23.get_shape().as_list()[2]

    pooled = tf.nn.avg_pool(conv4_23, [1,k,k,1], [1,1,1,1], padding='SAME')

    dense = tf.contrib.layers.flatten(pooled)

    logits = tf.contrib.layers.fully_connected(dense, 47, activation_fn=None, scope="fully_connected")

    return logits

if __name__ == "__main__":

    img = tf.placeholder(tf.float32, [None, 28, 28, 1])

    logits = ResNet101(img)
