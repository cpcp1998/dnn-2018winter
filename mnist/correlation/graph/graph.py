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

def plot(x, y):
  plt.scatter(x, y)
  plt.xlim(-np.nanmax(x)*0.1, np.nanmax(x)*1.1)
  plt.ylim(-np.nanmax(y)*0.1, np.nanmax(y)*1.1)

for layer in ['h_conv1', 'h_conv2', 'h_fc1', 'y_conv']:
  x = getattr(train0, layer)
  y = getattr(train1, layer)
  for i in range(x.shape[0]):
    plot(x[i], y[i])
    plt.savefig('./%s/%s_%d.png' % (layer, layer, i))
    plt.clf()
