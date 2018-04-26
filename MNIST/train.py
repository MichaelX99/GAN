import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#from model import *
#import os
#import numpy as np
#import time
#import imageio
from helper import *
from glob import glob

from input_pipeline import input_pipeline

# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 20
num_gpus = 3

# load MNIST
#mnist = input_data.read_data_sets("/home/mikep/DataSets/MNIST/", one_hot=True, reshape=[])

# Directory for the MNIST dataset
dataset_dir = '/home/mikep/DataSets/MNIST/tfrecord'

# Aquire all the sharded tfrecord files
dataset = glob(dataset_dir + "/*")
dataset.sort()

# Generate the input pipeline
imgs, noise, labels = input_pipeline(train_shards=dataset, batch_size=batch_size, num_preprocess_threads=4, num_gpus=num_gpus)
batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([imgs, noise, labels], capacity=2 * num_gpus)

tower_grads = []
with tf.variable_scope(tf.get_variable_scope()):
    for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                # Get this gpu's data
                image_batch, noise_batch, label_batch = batch_queue.dequeue()

                loss = tower_loss(image_batch, noise_batch, label_batch)

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                tf.get_variable_scope().reuse_variables()

                # Compute the gradients
                #grads = opt.compute_gradients(loss)

                #tower_grads.append(grads)


class Trainer():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        # Aquire all the sharded tfrecord files
        self.dataset = glob(dataset_dir + "/*")
        self.dataset.sort()

        # Form the optimizers
        lr = 0.0002
        self.D_optim = tf.train.AdamOptimizer(lr, beta1=0.5)
        self.G_optim = tf.train.AdamOptimizer(lr, beta1=0.5)

        self.num_gpus = 3
        self.batch_size = 100

    def form_input_pipeline(self):
        self.imgs, self.noise, self.labels = input_pipeline(train_shards=self.dataset,
                                                            batch_size=self.batch_size,
                                                            num_preprocess_threads=4,
                                                            num_gpus=self.num_gpus)

        self.batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([self.imgs, self.noise, self.labels], capacity=2 * self.num_gpus)

    def build_GAN(self):
        self.tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                        # Get this gpu's data
                        image_batch, noise_batch, label_batch = self.batch_queue.dequeue()

                        # Create the generator graph
                        G_z = generator(noise_batch)

                        # Create the discriminator graphs
                        D_real, D_real_logits = discriminator(image_batch)
                        D_fake, D_fake_logits = discriminator(G_z, reuse=True)

                        generator_loss, discriminator_loss = self.compute_loss(G_z, D_real, D_real_logits, D_fake, D_fake_logits, label_batch)


    def compute_loss(self, G_z, D_real, D_real_logits, D_fake, D_fake_logits, label_batch):
        pass

"""
# variables : input
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.Session()

#sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#tf.global_variables_initializer().run()

# MNIST resize and normalization
train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval(session=sess)
#train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# results save folder
root = 'MNIST_DCGAN_results/'
model = 'MNIST_DCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(mnist.train.num_examples // batch_size):
        # update discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result(sess, G_z, z, isTrain, (epoch + 1), save=True, path=fixed_p)
    #show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
"""
