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

class Nonzero(object):
  def __init__(self, activation):
    for layer in ['h_conv1', 'h_conv2', 'h_fc1', 'y_conv']:
      t = getattr(activation, layer)
      t = np.less(t, 1e-9)
      t = t.astype(int)
      setattr(self, layer, np.sum(t, axis=1))

nonzero0 = Nonzero(train0)

def plot(corr, nonzero):
  plt.scatter(corr, nonzero)
  plt.xlim(0, 1)
  plt.ylim(0, np.nanmax(nonzero)*1.1)

for layer in ['conv1', 'conv2', 'fc1']:
  correlation = getattr(corr, 'h_'+layer)
  nonzero = getattr(nonzero0, 'h_'+layer)
  plot(correlation, nonzero)
  plt.savefig('corr_nonzero_'+layer+'.png')
  plt.clf()
