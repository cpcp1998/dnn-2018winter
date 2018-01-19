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
networks = [untrained, train0, train1]
untrained.name = 'untrained'
train0.name = 'train0'
train1.name = 'train1'

for net1 in networks:
  for net2 in networks:
    if net1 != net2:
      for layer in layers:
        x = getattr(net1, layer)
        y = getattr(net2, layer)
        co = corrcoef(x, y)
        plt.hist(co, bins='auto', range=(0,1))
        plt.savefig('multi_%s_%s_to_%s.png' % (layer, net1.name, net2.name))
        plt.clf()

for i in range(4):
  layer = layers[i]
  unt = getattr(untrained, layer)
  t0 = getattr(train0, layer)
  t1 = getattr(train1, layer)
  co_unt = corrcoef(unt, t1)
  co_trd = corrcoef(t0, t1)
  plt.hist(co_trd, bins='auto', range=(0, 1), color='#0000FF80', density=True)
  plt.hist(co_unt, bins='auto', range=(0, 1), color='#FF000080', density=True)
  plt.xlabel('correlation r', fontsize=18)
  plt.ylabel('proportion of neurons/Δr', fontsize=18)
  plt.title(['conv1', 'conv2', 'fc1', 'fc2'][i], fontsize=18)
  plt.savefig('paper_multi_'+layer+'.png')
  plt.clf()
