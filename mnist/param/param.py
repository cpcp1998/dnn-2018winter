from tensorflow.python.keras.datasets.mnist import load_data

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
  init = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(init)

def bias_variable(shape):
  init = tf.constant(0.1, shape=shape)
  return tf.Variable(init)

(x_train, y_train), (x_test, y_test) = load_data()

x = tf.placeholder(tf.float32, [None, 28, 28])

y = tf.placeholder(tf.int64, [None])

with tf.name_scope('reshape'):
  x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

with tf.name_scope('pool1'):
  h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2'):
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

with tf.name_scope('pool2'):
  h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('loss'):
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      labels=y, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
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
      for i in range(8):
        param = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2][i]
        name = ['W_conv1', 'b_conv1', 'W_conv2', 'b_conv2', 'W_fc1', 'b_fc1', 'W_fc2', 'b_fc2'][i]
        param = sess.run(param)
        setattr(self, name, param)

untrained = Param('untrained')
train0 = Param('train0')
train1 = Param('train1')

for layer in ['W_conv1', 'b_conv1', 'W_conv2', 'b_conv2', 'W_fc1', 'b_fc1', 'W_fc2', 'b_fc2']:
  unt = getattr(untrained, layer)
  t0 = getattr(train0, layer)
  t1 = getattr(train1, layer)
  same = (t0 > unt) == (t1 > unt)
  same = np.average(same.astype(np.int32))
  print('%g of parameters changed in the same direction in layer %s' % (same, layer))
  unt_mean = np.average(np.fabs(unt))
  change_mean = np.average(np.fabs(t0 - unt))
  print('Parameters of layer %s changed by %g' % (layer, change_mean/unt_mean))
  norm1 = np.sqrt(np.vdot(t0 - unt, t0 - unt))
  norm2 = np.sqrt(np.vdot(t1 - unt, t1 - unt))
  dot = np.vdot(t0 - unt, t1 - unt)
  print('cos of angle between two changes of layer %s: %g' % (layer, dot/norm1/norm2))
  plt.hist((t0 - unt).reshape([-1]), bins='auto')
  plt.savefig('param_chg_%s.png' % layer)
  plt.clf()
  plt.hist(unt.reshape([-1]), bins='auto')
  plt.savefig('param_init_%s.png' % layer)
  plt.clf()
  plt.hist(t0.reshape([-1]), bins='auto')
  plt.savefig('param_final_%s.png' % layer)
  plt.clf()
