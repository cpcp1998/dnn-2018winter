import sys
sys.path.append("..")

from model import model
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Activation(object):
  def __init__(self, name):
    with tf.Session(graph=model.graph) as sess:
      model.load(sess, '../model/'+name+'.ckpt')
      h_conv1, h_conv2, h_fc1, y_conv = sess.run(
              (model.h_conv1, model.h_conv2, model.h_fc1, model.y_conv),
              feed_dict={model.x: model.x_test, model.keep_prob: 1.0})
      self.h_conv1 = np.transpose(h_conv1, (3, 0, 1, 2)).reshape((32, 10000*28*28))
      self.h_conv2 = np.transpose(h_conv2, (3, 0, 1, 2)).reshape((64, 10000*14*14))
      self.h_fc1 = np.transpose(h_fc1, (1, 0))
      self.y_conv = np.transpose(y_conv, (1, 0))

untrained = Activation('untrained')
train0 = Activation('train0')
train1 = Activation('train1')

def corrcoef(x, y):
  dim = x.shape[0]
  corr = np.corrcoef(x, y)
  return corr[:dim, dim:]

class Corrcoef(object):
  def __init__(self, x, y):
    self.h_conv1 = corrcoef(x.h_conv1, y.h_conv1)
    self.h_conv2 = corrcoef(x.h_conv2, y.h_conv2)
    self.h_fc1 = corrcoef(x.h_fc1, y.h_fc1)
    self.y_conv = corrcoef(x.y_conv, y.y_conv)

def absmax(array):
  array = np.fabs(array)
  return np.nanmax(array, axis=1)

def plot(array):
  array = absmax(array)
  plt.hist(array, bins='auto', range=(0, 1))

def plot_and_save(corr, name):
  plot(corr.h_conv1)
  plt.savefig(name+'+h_conv1.png')
  plt.clf()
  print('%s h_conv1: %g' % (name, np.nanmean(absmax(corr.h_conv1))))
  plot(corr.h_conv2)
  plt.savefig(name+'+h_conv2.png')
  plt.clf()
  print('%s h_conv2: %g' % (name, np.nanmean(absmax(corr.h_conv2))))
  plot(corr.h_fc1)
  plt.savefig(name+'+h_fc1.png')
  plt.clf()
  print('%s h_fc1: %g' % (name, np.nanmean(absmax(corr.h_fc1))))
  plot(corr.y_conv)
  plt.savefig(name+'+y_conv.png')
  plt.clf()
  print('%s y_conv: %g' % (name, np.nanmean(absmax(corr.y_conv))))

untrain = Corrcoef(train0, untrained)
plot_and_save(untrain, 'untrain')
train = Corrcoef(train0, train1)
plot_and_save(train, 'train')
untrain_rev = Corrcoef(untrained, train1)
plot_and_save(untrain_rev, 'untrain_rev')

for i in range(4):
  layer = ['h_conv1', 'h_conv2', 'h_fc1', 'y_conv'][i]
  unt = absmax(getattr(untrain_rev, layer))
  trd = absmax(getattr(train, layer))
  plt.hist(trd, bins='auto', range=(0, 1), color='#0000FF80', density=True)
  plt.hist(unt, bins='auto', range=(0, 1), color='#FF000080', density=True)
  plt.xlabel('correlation r')
  plt.ylabel('proportion of neurons/Δr')
  plt.title(['conv1', 'conv2', 'fc1', 'fc2'][i])
  plt.savefig('paper_'+layer+'.png')
  plt.clf()
