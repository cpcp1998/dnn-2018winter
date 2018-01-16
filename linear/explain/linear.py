import sys
sys.path.append("../..")

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
      model.load(sess, '../../model/'+name+'.ckpt')
      h_conv1, h_conv2, h_fc1, y_conv = sess.run(
              (model.h_conv1, model.h_conv2, model.h_fc1, model.y_conv),
              feed_dict={model.x: model.x_test, model.keep_prob: 1.0})
      self.h_conv1 = np.transpose(h_conv1, (3, 0, 1, 2)).reshape((32, 10000*28*28))
      self.h_conv2 = np.transpose(h_conv2, (3, 0, 1, 2)).reshape((64, 10000*14*14))
      self.h_fc1 = np.transpose(h_fc1, (1, 0))
      self.y_conv = np.transpose(y_conv, (1, 0))

train0 = Activation('train0')
train1 = Activation('train1')

def multivar_linear_loss(x, y):
  x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)
  yyt = np.vdot(y,y)
  xt = np.transpose(x)
  yxt = np.matmul(y, xt)
  xxt = np.matmul(x, xt)
  xyt = np.matmul(x, y)
  xxti = np.linalg.pinv(xxt)
  temp = np.matmul(yxt, xxti)
  return yyt - np.vdot(temp, xyt)

def linear_similarity(x, y):
  x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)
  xt = np.transpose(x)
  yt = np.transpose(y)
  yyt = np.matmul(y, yt)
  yxt = np.matmul(y, xt)
  xxt = np.matmul(x, xt)
  xyt = np.matmul(x, yt)
  xxti = np.linalg.pinv(xxt)
  temp = np.matmul(yxt, xxti)
  temp = yyt - np.matmul(temp, xyt)
  return np.diag(temp) / y.shape[1]

def corrcoef_one(x, y):
  std = np.std(y)
  return np.divide(multivar_linear_loss(x, y), std*std)

def corrcoef(x, y):
  num = linear_similarity(x, y)
  std = np.std(y, axis = 1)
  temp = 1 - np.divide(num, np.square(std))
  return np.sqrt(np.fabs(temp))

layers = ['h_conv1', 'h_conv2', 'h_fc1', 'y_conv']
train0.name = 'train0'
train1.name = 'train1'

def significance(network, layer):
  with tf.Session(graph=model.graph) as sess:
    sess.run(tf.global_variables_initializer())
    model.load(sess, '../../model/'+network+'.ckpt')
    layer = getattr(model, layer)
    grad = tf.gradients(model.cross_entropy, layer)
    grad = sess.run(grad, feed_dict={model.x: model.x_test, model.y: model.y_test, model.keep_prob: 1.0})
    grad =  np.fabs(grad)
    grad = np.sum(grad, (0,1,2,3)[:grad.ndim-1])
    return grad

for layer in layers:
  x = getattr(train0, layer)
  y = getattr(train1, layer)
  co = corrcoef(x, y)
  sig = significance(train1.name, layer)
  plt.scatter(co, sig)
  plt.xlim(0, 1)
  plt.ylim(0, np.nanmax(sig)*1.1)
  plt.savefig('multi_%s.png' % layer)
  plt.clf()

