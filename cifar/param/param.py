from tensorflow.python.keras.datasets.cifar10 import load_data

import tensorflow as tf
import numpy as np
import random as rn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  init = tf.glorot_normal_initializer()
  return tf.get_variable(name='Variable', initializer=init, shape=shape, dtype=tf.float32)

def bias_variable(shape):
  init = tf.constant(0, shape=shape, dtype=tf.float32)
  return tf.Variable(init)

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np.reshape(y_train, [50000])
y_test = np.reshape(y_test, [10000])

x = tf.placeholder(tf.float32, [None, 32, 32, 3])

y = tf.placeholder(tf.int64, [None, ])

with tf.variable_scope('dropout'):
  keep_prob1 = tf.placeholder(tf.float32)
  keep_prob2 = tf.placeholder(tf.float32)

with tf.variable_scope('reshape'):
  x_image = tf.reshape(x, [-1, 32, 32, 3])

with tf.variable_scope('conv1'):
  W_conv1 = weight_variable([3, 3, 3, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tf.variable_scope('conv2'):
  W_conv2 = weight_variable([3, 3, 32, 32])
  b_conv2 = bias_variable([32])
  h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

with tf.variable_scope('pool1'):
  h_pool1 = max_pool_2x2(h_conv2)

with tf.variable_scope('dropout1'):
  h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob1)

with tf.variable_scope('conv3'):
  W_conv3 = weight_variable([3, 3, 32, 64])
  b_conv3 = bias_variable([64])
  h_conv3 = tf.nn.relu(conv2d(h_pool1_drop, W_conv3) + b_conv3)

with tf.variable_scope('conv4'):
  W_conv4 = weight_variable([3, 3, 64, 64])
  b_conv4 = bias_variable([64])
  h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

with tf.variable_scope('pool2'):
  h_pool2 = max_pool_2x2(h_conv4)

with tf.variable_scope('dropout2'):
  h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob1)

with tf.variable_scope('fc1'):
  W_fc1 = weight_variable([8 * 8 * 64, 512])
  b_fc1 = bias_variable([512])

  h_pool2_flat = tf.reshape(h_pool2_drop, [-1, 8 * 8 * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.variable_scope('dropout3'):
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)

with tf.variable_scope('fc2'):
  W_fc2 = weight_variable([512, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.variable_scope('loss'):
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=y, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.variable_scope('optimizer'):
#  train_step = tf.train.RMSPropOptimizer(1e-4, 1e-6).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.variable_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), y)
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph = tf.get_default_graph()
saver = tf.train.Saver()

def load(sess, path):
  saver.restore(sess, path)
  print('Model restored from %s' % path)

class Param(object):
  def __init__(self, name):
    with tf.Session() as sess:
      load(sess, '../model/'+name+'.ckpt')
      for i in range(12):
        param = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, W_fc2, b_fc2][i]
        name = ['W_conv1', 'b_conv1', 'W_conv2', 'b_conv2', 'W_conv3', 'b_conv3', 'W_conv4', 'b_conv4', 'W_fc1', 'b_fc1', 'W_fc2', 'b_fc2'][i]
        param = sess.run(param)
        setattr(self, name, param)

untrained = Param('untrained')
train0 = Param('train0')
train1 = Param('train1')

for layer in ['W_conv1', 'b_conv1', 'W_conv2', 'b_conv2', 'W_conv3', 'b_conv3', 'W_conv4', 'b_conv4', 'W_fc1', 'b_fc1', 'W_fc2', 'b_fc2']:
  unt = getattr(untrained, layer)
  t0 = getattr(train0, layer)
  t1 = getattr(train1, layer)
  same = (t0 > unt) == (t1 > unt)
  same = np.average(same.astype(np.int32))
  print('%g of parameters changed in the same direction in layer %s' % (same, layer))
  unt_mean = np.average(np.fabs(unt))
  change_mean = np.average(np.fabs(t0 - unt))
  print('Parameters of layer %s changed by %g' % (layer, change_mean/unt_mean))
  plt.hist((t0 - unt).reshape([-1]), bins='auto')
  plt.savefig('param_chg_%s.png' % layer)
  plt.clf()
  plt.hist(unt.reshape([-1]), bins='auto')
  plt.savefig('param_init_%s.png' % layer)
  plt.clf()
  plt.hist(t0.reshape([-1]), bins='auto')
  plt.savefig('param_final_%s.png' % layer)
  plt.clf()
