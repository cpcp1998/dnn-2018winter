import sys
sys.path.append("..")

from model import model
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

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

def multivar_linear_loss(x, y):
  yyt = np.vdot(y,y)
  xt = np.transpose(x)
  yxt = np.matmul(y, xt)
  xxt = np.matmul(x, xt)
  xyt = np.matmul(x, y)
  xxti = np.linalg.pinv(xxt)
  temp = np.matmul(yxt, xxti)
  return yyt - np.vdot(temp, xyt)

def linear_similarity(x, y):
  xt = np.transpose(x)
  yt = np.transpose(y)
  yyt = np.matmul(y, yt)
  yxt = np.matmul(y, xt)
  xxt = np.matmul(x, xt)
  xyt = np.matmul(x, yt)
  xxti = np.linalg.pinv(xxt)
  temp = np.matmul(yxt, xxti)
  temp = yyt - np.matmul(temp, xyt)
  return np.trace(temp) / x.shape[0] / x.shape[1]

layers = ['h_conv1', 'h_conv2', 'h_fc1', 'y_conv']
networks = [untrained, train0, train1]
untrained.name = 'untrained'
train0.name = 'train0'
train1.name = 'train1'

for net1 in networks:
  for net2 in networks:
    if net1 != net2:
      for layer in layers:
        print('Fit layer %s in %s using %s: %g' %
            (layer, net2.name, net1.name,
             math.sqrt(linear_similarity(getattr(net1,layer),
                                         getattr(net2, layer)))
            ))

