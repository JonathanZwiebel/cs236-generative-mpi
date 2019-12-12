#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Network definitions for multiplane image (MPI) prediction networks.
"""
from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
# from libs.ops import *

def mpi_net(inputs, num_outputs, ngf=64, vscope='net', reuse_weights=False):
  """Network definition for multiplane image (MPI) inference.

  Args:
    inputs: stack of input images [batch, height, width, input_channels]
    num_outputs: number of output channels
    ngf: number of features for the first conv layer
    vscope: variable scope
    reuse_weights: whether to reuse weights (for weight sharing)
  Returns:
    pred: network output at the same spatial resolution as the inputs.
  """
  with tf.variable_scope(vscope, reuse=reuse_weights):
#     with tf.device("/device:XLA_GPU:0"):
    with slim.arg_scope(
    [slim.conv2d, slim.conv2d_transpose], normalizer_fn=slim.layer_norm):
        cnv1_1 = slim.conv2d(inputs, ngf, [3, 3], scope='conv1_1', stride=1)
        cnv1_2 = slim.conv2d(cnv1_1, ngf * 2, [3, 3], scope='conv1_2', stride=2)

        cnv2_1 = slim.conv2d(cnv1_2, ngf * 2, [3, 3], scope='conv2_1', stride=1)
        cnv2_2 = slim.conv2d(cnv2_1, ngf * 4, [3, 3], scope='conv2_2', stride=2)

        cnv3_1 = slim.conv2d(cnv2_2, ngf * 4, [3, 3], scope='conv3_1', stride=1)
        cnv3_2 = slim.conv2d(cnv3_1, ngf * 4, [3, 3], scope='conv3_2', stride=1)
        cnv3_3 = slim.conv2d(cnv3_2, ngf * 8, [3, 3], scope='conv3_3', stride=2)

        cnv4_1 = slim.conv2d(
          cnv3_3, ngf * 8, [3, 3], scope='conv4_1', stride=1, rate=2)
        cnv4_2 = slim.conv2d(
          cnv4_1, ngf * 8, [3, 3], scope='conv4_2', stride=1, rate=2)
        cnv4_3 = slim.conv2d(
          cnv4_2, ngf * 8, [3, 3], scope='conv4_3', stride=1, rate=2)

        # Adding skips
        skip = tf.concat([cnv4_3, cnv3_3], axis=3)
        cnv6_1 = slim.conv2d_transpose(
          skip, ngf * 4, [4, 4], scope='conv6_1', stride=2)
        cnv6_2 = slim.conv2d(cnv6_1, ngf * 4, [3, 3], scope='conv6_2', stride=1)
        cnv6_3 = slim.conv2d(cnv6_2, ngf * 4, [3, 3], scope='conv6_3', stride=1)

        skip = tf.concat([cnv6_3, cnv2_2], axis=3)
        cnv7_1 = slim.conv2d_transpose(
          skip, ngf * 2, [4, 4], scope='conv7_1', stride=2)
        cnv7_2 = slim.conv2d(cnv7_1, ngf * 2, [3, 3], scope='conv7_2', stride=1)

        skip = tf.concat([cnv7_2, cnv1_2], axis=3)
        cnv8_1 = slim.conv2d_transpose(
          skip, ngf, [4, 4], scope='conv8_1', stride=2)
        cnv8_2 = slim.conv2d(cnv8_1, ngf, [3, 3], scope='conv8_2', stride=1)

        feat = cnv8_2

        pred = slim.conv2d(
          feat,
          num_outputs, [1, 1],
          stride=1,
          activation_fn=tf.nn.tanh,
          normalizer_fn=None,
          scope='color_pred')
    return pred

# # just append to MPI net???, MPI net has threee outputs -- only apply GAN to one output
# # train GAN in particular way, cannot just append???
# class DCGANGenerator(object):

#   def __init__(self, hidden_dim=128, batch_size=64, hidden_activation=tf.nn.relu, output_activation=tf.nn.tanh, use_batch_norm=True, z_distribution='normal', scope='generator', **kwargs):
#     self.hidden_dim = hidden_dim
#     self.batch_size = batch_size
#     self.hidden_activation = hidden_activation
#     self.output_activation = output_activation
#     self.use_batch_norm = use_batch_norm
#     self.z_distribution = z_distribution
#     self.scope = scope

#   def __call__(self, z, is_training=True, **kwargs):
#     with tf.variable_scope(self.scope):
#       if self.use_batch_norm:
#         l0  = self.hidden_activation(batch_norm(linear(z, 4 * 4 * 512, name='l0', stddev=0.02), name='bn0', is_training=is_training))
#         l0  = tf.reshape(l0, [self.batch_size, 4, 4, 512])
#         dc1 = self.hidden_activation(batch_norm(deconv2d( l0, [self.batch_size,  8,  8, 256], name='dc1', stddev=0.02), name='bn1', is_training=is_training))
#         dc2 = self.hidden_activation(batch_norm(deconv2d(dc1, [self.batch_size, 16, 16, 128], name='dc2', stddev=0.02), name='bn2', is_training=is_training))
#         dc3 = self.hidden_activation(batch_norm(deconv2d(dc2, [self.batch_size, 32, 32,  64], name='dc3', stddev=0.02), name='bn3', is_training=is_training))
#         dc4 = self.output_activation(deconv2d(dc3, [self.batch_size, 32, 32, 3], 3, 3, 1, 1, name='dc4', stddev=0.02))
#       else:
#         l0  = self.hidden_activation(linear(z, 4 * 4 * 512, name='l0', stddev=0.02))
#         l0  = tf.reshape(l0, [self.batch_size, 4, 4, 512])
#         dc1 = self.hidden_activation(deconv2d(l0, [self.batch_size, 8, 8, 256], name='dc1', stddev=0.02))
#         dc2 = self.hidden_activation(deconv2d(dc1, [self.batch_size, 16, 16, 128], name='dc2', stddev=0.02))
#         dc3 = self.hidden_activation(deconv2d(dc2, [self.batch_size, 32, 32, 64], name='dc3', stddev=0.02))
#         dc4 = self.output_activation(deconv2d(dc3, [self.batch_size, 32, 32, 3], 3, 3, 1, 1, name='dc4', stddev=0.02))
#       x = dc4
#     return x

#   def generate_noise(self):
#     if self.z_distribution == 'normal':
#       return np.random.randn(self.batch_size, self.hidden_dim).astype(np.float32)
#     elif self.z_distribution == 'uniform' :
#       return np.random.uniform(-1, 1, (self.batch_size, self.hidden_dim)).astype(np.float32)
#     else:
#       raise NotImplementedError


# class SNDCGAN_Discrminator(object):

#   def __init__(self, batch_size=64, hidden_activation=lrelu, output_dim=1, scope='critic', **kwargs):
#     self.batch_size = batch_size
#     self.hidden_activation = hidden_activation
#     self.output_dim = output_dim
#     self.scope = scope

#   def __call__(self, x, update_collection=tf.GraphKeys.UPDATE_OPS, **kwargs):
#     with tf.variable_scope(self.scope):
#       c0_0 = self.hidden_activation(conv2d(   x,  64, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_0'))
#       c0_1 = self.hidden_activation(conv2d(c0_0, 128, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_1'))
#       c1_0 = self.hidden_activation(conv2d(c0_1, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_0'))
#       c1_1 = self.hidden_activation(conv2d(c1_0, 256, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_1'))
#       c2_0 = self.hidden_activation(conv2d(c1_1, 256, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_0'))
#       c2_1 = self.hidden_activation(conv2d(c2_0, 512, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_1'))
#       c3_0 = self.hidden_activation(conv2d(c2_1, 512, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c3_0'))
#       c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
#       l4 = linear(c3_0, self.output_dim, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='l4')
#     return tf.reshape(l4, [-1])
