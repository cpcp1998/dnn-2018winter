import sys
sys.path.append("../..")

from model import model
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Activation(object):
  def __init__(self, name):
    with tf.Session(graph=model.graph) as sess:
      model.load(sess, '../../model/'+name+'.ckpt')
      h_conv1, h_conv2, h_fc1, y_conv = sess.run(
              (model.h_conv1, model.h_conv2, model.h_fc1, model.y_conv),
              feed_dict={model.x: model.x_test, model.keep_prob: 1.0})
      self.h_conv1 = np.transpose(h_conv1, (3, 0, 1, 2)).reshape((32, 10000*28*28))
      self.h_conv2 = np.transpose(h_conv2, (3, 0, 1, 2)).reshape((64, 10000*14*14))
      self.h_fc1 = np.transpose(h_fc1, (1, 0))
      self.y_conv = np.transpose(y_conv, (1, 0))

#untrained = Activation('untrained')
train0 = Activation('train0')
train1 = Activation('train1')

def corrcoef(x, y):
  dim = x.shape[0]
  corr = np.corrcoef(x, y)
  return corr[:dim, dim:]

def absmax(array):
  array = np.fabs(array)
  return np.nanmax(array, axis=1)

class Corr(object):
  def __init__(self, x, y):
    self.h_conv1 = absmax(corrcoef(x.h_conv1, y.h_conv1))
    self.h_conv2 = absmax(corrcoef(x.h_conv2, y.h_conv2))
    self.h_fc1 = absmax(corrcoef(x.h_fc1, y.h_fc1))
    self.y_conv = absmax(corrcoef(x.y_conv, y.y_conv))

corr = Corr(train0, train1)

class Weight(object):
  def __init__(self, name):
    with tf.Session(graph=model.graph) as sess:
      model.load(sess, '../../model/'+name+'.ckpt')
      W_conv2, W_fc1, W_fc2 = sess.run(
              (model.W_conv2, model.W_fc1, model.W_fc2))
      W_conv2 = np.square(W_conv2)
      W_fc1 = np.square(W_fc1)
      W_fc2 = np.square(W_fc2)
      W_conv2 = np.transpose(W_conv2, (2, 0, 1, 3)).reshape((32, 5*5*64))
      W_fc1 = W_fc1.reshape((7, 7, 64, 1024))
      W_fc1 = np.transpose(W_fc1, (2, 0, 1, 3)).reshape((64, 7*7*1024))
      self.W_conv1 = np.sum(W_conv2, axis=1)
      self.W_conv2 = np.sum(W_fc1, axis=1)
      self.W_fc1 = np.sum(W_fc2, axis=1)

weight0 = Weight('train0')

def plot(corr, weight):
  plt.scatter(corr, weight)
  plt.xlim(0, 1)
  plt.ylim(0, np.nanmax(weight)*1.1)

for layer in ['conv1', 'conv2', 'fc1']:
  correlation = getattr(corr, 'h_'+layer)
  weight = getattr(weight0, 'W_'+layer)
  plot(correlation, weight)
  plt.savefig('corr_sqrweight_'+layer+'.png')
  plt.clf()
