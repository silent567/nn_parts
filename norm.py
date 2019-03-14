#!/usr/bin/env python
# coding=utf-8

# 20180807 by tanghao
# reorganized from nn_parts.py
# This file contains normalization-related layers

import tensorflow as tf
from .init_var import *

class BatchNorm:
    def __init__(self,channel_size,name_scope='BatchNorm',summ_flag=True,
                 offset=None,offset_init=None,scale=None,scale_init=None):
        if not isinstance(channel_size,list):
            channel_size = [channel_size,]
        self.channel_size= [int(c) for c in channel_size]
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            with tf.name_scope('initialize_variables'):
                self.offset = init_zero_variable(offset,offset_init,self.channel_size,'offset')
                self.scale = init_one_variable(scale,scale_init,self.channel_size,'scale')
                if self.summ_flag:
                    self.offset_summ = tf.summary.histogram('offset',self.offset)
                    self.scale_summ = tf.summary.histogram('scale',self.scale)
    def __call__(self,input_tensor):
        return self.get_output(input_tensor)
    def get_output(self,input_tensor):
        with tf.name_scope(self.name_scope):
            mean,var = tf.nn.moments(input_tensor,list(range(len(input_tensor.shape)-len(self.channel_size))),name='moments')
            output_tensor = tf.nn.batch_normalization(input_tensor,mean,var,self.offset,self.scale,1e-7,name='output_tensor')
        return output_tensor

class LayerNorm:
    def __init__(self,channel_size,axes=None,name_scope='LayerNorm',summ_flag=True,
                 offset=None,offset_init=None,scale=None,scale_init=None):
        if not isinstance(channel_size,list):
            channel_size = [channel_size,]
        self.channel_size= [int(c) if not isinstance(c,tf.Tensor) else c for c in channel_size]
        self.axes = axes
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            with tf.name_scope('initialize_variables'):
                self.offset = init_zero_variable(offset,offset_init,self.channel_size,'offset')
                self.scale = init_one_variable(scale,scale_init,self.channel_size,'scale')
                if self.summ_flag:
                    self.offset_summ = tf.summary.histogram('offset',self.offset)
                    self.scale_summ = tf.summary.histogram('scale',self.scale)
    def __call__(self,input_tensor):
        return self.get_output(input_tensor)
    def get_output(self,input_tensor):
        with tf.name_scope(self.name_scope):
            if self.axes is None:
                axes = list(range(len(input_tensor.shape)-len(self.channel_size),len(input_tensor.shape)))
            else:
                axes = self.axes
            mean,var = tf.nn.moments(input_tensor,axes,name='moments',keep_dims=True)
            output_tensor = tf.nn.batch_normalization(input_tensor,mean,var,self.offset,self.scale,1e-7,name='output_tensor')
        return output_tensor

class _LayerNorm_old_version: # old version of layernorm
    def __init__(self,layer_dim,name_scope='LayerNorm',summ_flag=True,
                 gain=None,gain_init=None,bias=None,bias_init=None):
        if type(layer_dim) != list:
            self.shape = [layer_dim,]
        else:
            self.shape = layer_dim
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            with tf.name_scope('initialize_variables'):
                self.gain = init_one_variable(gain,gain_init,self.shape,'gain')
                self.bias = init_zero_variable(bias,bias_init,self.shape,'bias')
        if self.summ_flag:
            self.gain_summ = tf.summary.histogram('gain',self.gain)
            self.bias_summ = tf.summary.histogram('bias',self.bias)
    def __del__(self):
        del self.shape
        del self.gain
        del self.bias
        if self.summ_flag:
            del self.gain_summ
            del self.bias_summ
        del self.summ_flag
    def __call__(self,input_tensor,axis=-1):
        return self.get_output(input_tensor,axis)
    def get_output(self,input_tensor,axis=-1):
        with tf.name_scope(self.name_scope):
            mean = tf.reduce_mean(input_tensor,axis=axis,keepdims=True,name='mean')
            std = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(input_tensor,mean)),axis=axis,keepdims=True),name='std')
            output_tensor = tf.add(tf.multiply(tf.divide(tf.subtract(input_tensor,mean),std),self.gain),self.bias,name='output_tensor')
        return output_tensor

if __name__ == '__main__':
    a = tf.random_normal([3,4,5],mean=1,stddev=2)
    norm_layer = LayerNorm([5])
    b = norm_layer(a)
    b_mean,b_var = tf.nn.moments(b,axes=[2])
    print(a.shape,b.shape,b_mean.shape,b_var.shape)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run([a,b,b_mean,b_var]))
