from tensorflow.python.keras.datasets.cifar10 import load_data

import tensorflow as tf
import numpy as np
import random as rn

def init_seed():
  np.random.seed(42)
  rn.seed(42)
  tf.set_random_seed(42)
init_seed()

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
  optimizer = tf.train.AdamOptimizer(1e-4)
  train_conv1 = optimizer.minimize(cross_entropy, var_list=[W_conv1, b_conv1])
  train_conv2 = optimizer.minimize(cross_entropy, var_list=[W_conv2, b_conv2])
  train_conv3 = optimizer.minimize(cross_entropy, var_list=[W_conv3, b_conv3])
  train_conv4 = optimizer.minimize(cross_entropy, var_list=[W_conv4, b_conv4])
  train_fc1 = optimizer.minimize(cross_entropy, var_list=[W_fc1, b_fc1])
  train_fc2 = optimizer.minimize(cross_entropy, var_list=[W_fc2, b_fc2])

with tf.variable_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), y)
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

saver_conv1 = tf.train.Saver({
    W_conv1.name[:-2]: W_conv1,
    b_conv1.name[:-2]: b_conv1,
})

saver_conv2 = tf.train.Saver({
    W_conv2.name[:-2]: W_conv2,
    b_conv2.name[:-2]: b_conv2,
})

saver_conv3 = tf.train.Saver({
    W_conv3.name[:-2]: W_conv3,
    b_conv3.name[:-2]: b_conv3,
})

saver_conv4 = tf.train.Saver({
    W_conv4.name[:-2]: W_conv4,
    b_conv4.name[:-2]: b_conv4,
})

saver_fc1 = tf.train.Saver({
    W_fc1.name[:-2]: W_fc1,
    b_fc1.name[:-2]: b_fc1,
})

saver_fc2 = tf.train.Saver({
    W_fc2.name[:-2]: W_fc2,
    b_fc2.name[:-2]: b_fc2,
})

savers = [saver_conv1, saver_conv2, saver_conv3, saver_conv4, saver_fc1, saver_fc2]
train_steps = [train_conv1, train_conv2, train_conv3, train_conv4, train_fc1, train_fc2]

def train(sess, train_step, lower_bound, upper_bound, batch_size, epoch):
#  train_accuracy = accuracy.eval(session=sess, feed_dict={
#      x: x_test, y: y_test, keep_prob: 1.0})
#  print('init, test accuracy %g' % train_accuracy)
  for i in range(epoch):
    left = lower_bound
    right = lower_bound + batch_size
    while right <= upper_bound:
      train_step.run(session=sess, feed_dict={x: x_train[left:right], y: y_train[left:right], keep_prob1: 0.75, keep_prob2: 0.5})
      left += batch_size
      right += batch_size
    train_accuracy = accuracy.eval(session=sess, feed_dict={
        x: x_test, y: y_test, keep_prob1: 1.0, keep_prob2: 1.0})
#    print('epoch %d, test accuracy %g' % (i, train_accuracy))
#  print('test accuracy %g' % accuracy.eval(feed_dict={
#      x: x_test, y: y_test, keep_prob: 1.0}))

def similar_rate(low_model, up_model, layer, lower_bound, upper_bound):
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for saver in savers[:layer]:
      saver.restore(sess, '../model/'+low_model+'.ckpt')
    for saver in savers[layer:]:
      saver.restore(sess, '../model/'+up_model+'.ckpt')
    before = accuracy.eval(feed_dict={x: x_test, y: y_test, keep_prob1: 1.0, keep_prob2: 1.0})
    train(sess, train_steps[layer], lower_bound, upper_bound, 50, 50)
    after = accuracy.eval(feed_dict={x: x_test, y: y_test, keep_prob1: 1.0, keep_prob2: 1.0})
    return before, after

for upper in ['train0', 'train1']:
  lower_bound = 0
  upper_bound = 25000
  if upper == 'train1':
    lower_bound += 25000
    upper_bound += 25000
  for lower in ['untrained', 'train0', 'train1']:
    if lower != upper:
      for layer in [1, 2, 3, 4, 5]:
        before, after = similar_rate(lower, upper, layer, lower_bound, upper_bound)
        print('layer %s of %s feed into %s: %g, %g' % (
          ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2'][layer-1],
          lower, upper, before, after))

