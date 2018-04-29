import tensorflow as tf
from helper import *
from glob import glob
from input_pipeline import input_pipeline
from model import generator, discriminator
import matplotlib.pyplot as plt


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
        train_epoch = 20
        train_examples = 50000

        self.train_iters = (train_examples // (self.batch_size * self.num_gpus)) * train_epoch

    def form_input_pipeline(self):
        self.imgs, self.labels, self.noises = input_pipeline(train_shards=self.dataset,
                                                            batch_size=self.batch_size,
                                                            num_preprocess_threads=4,
                                                            num_gpus=self.num_gpus)

        self.batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([self.imgs, self.labels, self.noises], capacity=2 * self.num_gpus)

    def average_gradients(self, tower_grads):
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

      return average_grads

    def build_GAN(self):
        tower_D_grads = []
        tower_G_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                        # Get this gpu's data
                        image_batch, label_batch, noise_batch = self.batch_queue.dequeue()

                        # Create the generator graph
                        self.G_z = generator(noise_batch)

                        # Create the discriminator graphs
                        D_real, D_real_logits = discriminator(image_batch)
                        D_fake, D_fake_logits = discriminator(self.G_z, reuse=True)

                        self.compute_loss(self.G_z, D_real, D_real_logits, D_fake, D_fake_logits, label_batch)

                        D_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
                        G_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]

                        tf.get_variable_scope().reuse_variables()

                        D_grads = self.D_optim.compute_gradients(self.D_loss, var_list=D_vars)
                        tower_D_grads.append(D_grads)

                        G_grads = self.G_optim.compute_gradients(self.G_loss, var_list=G_vars)
                        tower_G_grads.append(G_grads)

        avg_D_grads = self.average_gradients(tower_D_grads)
        avg_G_grads = self.average_gradients(tower_G_grads)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.D_train_op = self.D_optim.apply_gradients(avg_D_grads)
            self.G_train_op = self.G_optim.apply_gradients(avg_G_grads)

            self.train_op = tf.group(self.D_train_op, self.G_train_op)


    def compute_loss(self, G_z, D_real, D_real_logits, D_fake, D_fake_logits, label_batch):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([self.batch_size, 1, 1, 1])))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([self.batch_size, 1, 1, 1])))

        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([self.batch_size, 1, 1, 1])))

    def save_imgs(self, time):
        print("saving images")

        imgs = self.sess.run(self.G_z)

        for i, img in enumerate(imgs):
            plt.figure()
            plt.imshow(np.reshape(img, (64, 64)), cmap='gray')
            path = 'imgs/' + str(time) + '/' + str(i) + '.png'
            plt.savefig(path)
            plt.close()

    def train(self):
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.allow_soft_placement=True
        config.log_device_placement=False

        self.sess = tf.Session(config=config)

        self.sess.run(init_op)
        tf.train.start_queue_runners(sess=self.sess)

        time = 0
        for i in range(self.train_iters):
            d_l, g_l, _ = self.sess.run([self.D_loss, self.G_loss, self.train_op])
            print("iter = "+str(i) + ", D = "+str(d_l) + ", G = "+str(g_l))
            if i % 500 == 0 and i != 0:
                self.save_imgs(time)
                time += 1

if __name__ == "__main__":
    # Directory for the MNIST dataset
    dataset_dir = '/home/mikep/DataSets/MNIST/tfrecord'

    trainer = Trainer(dataset_dir)
    trainer.form_input_pipeline()
    trainer.build_GAN()

    trainer.train()

"""

from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import time
import imageio
import pickle

# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 20
num_gpus = 3

# load MNIST
mnist = input_data.read_data_sets("/home/mikep/DataSets/MNIST/", one_hot=True, reshape=[])

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


        #loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        #D_losses.append(loss_d_)
        #
        ## update generator
        #z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        #loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
        #G_losses.append(loss_g_)

        loss_d_, _, loss_g_, _ = sess.run([D_loss, D_optim, G_loss, G_optim], {z: z_, x: x_, isTrain: True})

        print("iter = "+str(iter) + ", D = "+str(loss_d_) + ", G = "+str(loss_g_))

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
