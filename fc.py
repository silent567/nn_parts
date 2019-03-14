#!/usr/bin/env python
# coding=utf-8

# 20180805 by tanghao
# reorganized from nn_parts.py
# This file contains Fully-Connected-related layers

import tensorflow as tf
from .init_var import *
from .norm import LayerNorm as _LayerNorm

class Identity:
    def __init__(self,):
        pass
    def __call__(self,input_tensor):
        return self.get_output(input_tensor)
    def get_output(self,input_tensor):
        return input_tensor
    def get_l2_loss(self,):
        return tf.zeros([])

def linear_activation(x):
    return x

class Dense:
    '''
    The class for the Dense layer, which is a simply feed-forward layer
    The input neuron number is input_size, the output neuron number is output size.
    The input_tensor should be tensor of shape [N,input_size]
    W is the weight matrix and b is the bias vector.
    This layer performs: output_tensor = input_tensor * W + b
    '''
    def __init__(self,input_size,output_size,name_scope='Dense',activation_func=tf.nn.relu,W=None,W_init=None,b=None,b_init=None,summ_flag=True,bias_flag=True):
        '''
        The input_size and output_size is the number of input and output tensors
        name_scope should be of type string
        activation_func should be function with tf.Tensor as arguments and output tf.Tensor
        W is tf.Variable with shape equal to [input_size,output_size]
        b is tf.Variable with shape equal to [output_size,]
        init_W can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape [input_size,output_size]
        init_b can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape [output_size]
        summ_flag is boolean, indicating whether tensors are summarized
        bias_flag is boolean, indicating whether biases are added
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = activation_func
        self.summ_flag = summ_flag
        self.bias_flag = True if bias_flag is None else bias_flag
        with tf.name_scope(name_scope) as self.name_scope:
            self.W = init_random_variable(W,W_init,[input_size,output_size],2./input_size,'W')
            if self.bias_flag:
                self.b = init_zero_variable(b,b_init,[output_size],'b')
            if summ_flag:
                self.W_sum = tf.summary.histogram(name_scope+':W',self.W)
                if self.bias_flag:
                    self.b_sum = tf.summary.histogram(name_scope+':b',self.b)

    def __del__(self):
        del self.input_size
        del self.output_size
        del self.name_scope
        del self.activation_func
        del self.W
        if self.bias_flag:
            del self.b
        if self.summ_flag:
            del self.W_sum
            if self.bias_flag:
                del self.b_sum
        del self.summ_flag
        del self.bias_flag
    def __call__(self,input_tensor):
        '''
        input_tensor should be tf.Tensor with shape [...,input_size]
        return output_tensor of shaape [...,output_size]
        '''
        return self.get_output(input_tensor)
    def get_output(self,input_tensor):
        '''
        input_tensor should be tf.Tensor with shape [N,input_size]
        return output_tensor of shape [N,output_size]
        '''
        with tf.name_scope(self.name_scope):
            if input_tensor.shape[-1].value != self.input_size:
                print(self.input_size,input_tensor.shape[-1].value)
                print('Error in Dense: input_size should be %d, but receive %d'%(self.input_size,input_tensor.shape[-1].value))
                return None
            if (tf.shape(input_tensor).shape[0].value != 2):
                output_shape = tf.concat([tf.shape(input_tensor)[:-1],tf.constant([self.output_size])],axis=0)
                output_tensor = tf.add(tf.matmul(tf.reshape(input_tensor,[-1,self.input_size]),self.W),self.b if self.bias_flag else tf.zeros([]),name='output_tensor')
                output_tensor = tf.reshape(self.activation_func(output_tensor),output_shape)
            else:
                output_tensor = tf.add(tf.matmul(input_tensor,self.W),self.b if self.bias_flag else tf.zeros([]),name='output_tensor')
            return self.activation_func(output_tensor)
    def get_l2_loss(self,):
        with tf.name_scope(self.name_scope):
            return tf.reduce_sum(tf.square(self.W),name='L2_loss')

def _get_first_half_channel(x,name='first_half_channel'):
    x_shape = x.shape.as_list()
    rank = len(x_shape)
    channel_num = x_shape[-1]
    return tf.slice(x,[0]*rank,[-1]*(rank-1)+[int(channel_num/2)])

def _get_last_half_channel(x,name='last_half_channel'):
    x_shape = x.shape.as_list()
    rank = len(x_shape)
    channel_num = x_shape[-1]
    return tf.slice(x,[0]*(rank-1)+[int(channel_num/2)],[-1]*rank)

def DenseBlock(input_tensor,hidden_size,train_flag=None,gated_flag=False,ln_flag=True,dp_flag=False,dp=0.5,activation_func=tf.nn.relu,name_scope='DenseBlock'):
    with tf.name_scope(name_scope):
        dense_layer = Dense(input_tensor.shape.as_list()[-1],hidden_size*(2 if gated_flag else 1),activation_func=linear_activation,name_scope='DenseLayer')
        dense_out = dense_layer(input_tensor)
        if gated_flag:
            dense_out_candidate = _get_first_half_channel(dense_out)
            if ln_flag:
                dense_candidate_ln_layer = _LayerNorm([hidden_size],name_scope='dense_candidate_layer_norm_layer')
                dense_out_candidate = dense_candidate_ln_layer(dense_out_candidate)
            dense_out_candidate = tf.tanh(dense_out_candidate)
            dense_out_gate = _get_last_half_channel(dense_out)
            if ln_flag:
                dense_gate_ln_layer = _LayerNorm([hidden_size],name_scope='dense_gate_layer_norm_layer')
                dense_out_gate = dense_gate_ln_layer(dense_out_gate)
            dense_out_gate = tf.sigmoid(dense_out_gate)
            dense_out = tf.multiply(dense_out_candidate,dense_out_gate,name='dense_gated_out')
        else:
            if ln_flag:
                dense_ln_layer = _LayerNorm([hidden_size],name_scope='dense_layer_norm_layer')
                dense_out = dense_ln_layer(dense_out)
            dense_out = activation_func(dense_out,name='dense_activated_out')
        if dp_flag:
            dense_out = tf.layers.dropout(dense_out,dp,training=train_flag,name='dense_out_drop')
    return dense_layer,dense_out

def stackedDenseBlock(input_tensor,hidden_size,layer_number,train_flag=None,gated_flag=False,ln_flag=True,dp_flag=False,dp=0.5,res_flag=True,skip_flag=True,activation_func=tf.nn.relu,name_scope='stackedDenseBlock'):
    with tf.name_scope(name_scope):
        stacked_dense_layers = []
        stacked_dense_input = [input_tensor]
        stacked_dense_output = []
        for layer_num in range(1,layer_number+1):
            dense_layer,dense_out = DenseBlock(stacked_dense_input[-1],hidden_size,train_flag=train_flag,gated_flag=gated_flag,ln_flag=ln_flag,dp_flag=dp_flag,dp=dp,activation_func=activation_func,name_scope='DenseBlock%d'%layer_num)
            stacked_dense_layers.append(dense_layer)
            stacked_dense_output.append(dense_out)
            if res_flag and layer_num > 1:
                stacked_dense_input.append(tf.add(stacked_dense_input[-1],dense_out,name='dense_residual_out%d'%layer_num))
            else:
                stacked_dense_input.append(dense_out)
        if skip_flag:
            stacked_dense_out = tf.add_n(stacked_dense_output,name='stacked_dense_out')
        else:
            stacked_dense_out = stacked_dense_input[-1]
    return stacked_dense_out, stacked_dense_layers,stacked_dense_output
