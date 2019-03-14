#!/usr/bin/env python
# coding=utf-8

# 20180805 by tanghao
# reorganized from nn_parts.py
# This file contains Conv-LSTM-related layers

import tensorflow as tf
from .init_var import *

class ConvLSTMCell:
    '''
    The class for the LSTM layer, where we replace traditional dense layer with 1D convolution
    The input neuron number is input_size, the output neuron number is output size. (the time step length is not needed)
    The input_tensor should be tensor of shape [N,T,input_size], where N is the batch size, T is the time step length.
    Wx_* are the weight matrices of shape [input_size,output_size], Wh_* are the weight matrices of shape [output_size,output_size] and b_* are the bias vector of shape [output_size].
    '''
    def __init__(self,input_size,input_channel,output_channel,kernel_size,name_scope='LSTMCell',summ_flag=True,
                 Fx_i=None,Fx_i_init=None,Fx_o=None,Fx_o_init=None,Fx_f=None,Fx_f_init=None,Fx_g=None,Fx_g_init=None,
                 Fh_i=None,Fh_i_init=None,Fh_o=None,Fh_o_init=None,Fh_f=None,Fh_f_init=None,Fh_g=None,Fh_g_init=None,
                 b_i=None,b_i_init=None,b_o=None,b_o_init=None,b_f=None,b_f_init=None,b_g=None,b_g_init=None):
        self.input_size = input_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            self.Fx_i = init_ortho_random_variable(Fx_i,Fx_i_init,[kernel_size,1,input_channel,output_channel],1./((input_channel+1)*kernel_size),self.name_scope+'Fx_i')
            self.Fx_o = init_ortho_random_variable(Fx_o,Fx_o_init,[kernel_size,1,input_channel,output_channel],1./((input_channel+1)*kernel_size),self.name_scope+'Fx_o')
            self.Fx_f = init_ortho_random_variable(Fx_f,Fx_f_init,[kernel_size,1,input_channel,output_channel],1./((input_channel+1)*kernel_size),self.name_scope+'Fx_f')
            self.Fx_g = init_ortho_random_variable(Fx_g,Fx_g_init,[kernel_size,1,input_channel,output_channel],1./((input_channel+1)*kernel_size),self.name_scope+'Fx_g')
            self.Fh_i = init_ortho_random_variable(Fh_i,Fh_i_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_i')
            self.Fh_o = init_ortho_random_variable(Fh_o,Fh_o_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_o')
            self.Fh_f = init_ortho_random_variable(Fh_f,Fh_f_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_f')
            self.Fh_g = init_ortho_random_variable(Fh_g,Fh_g_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_g')
            self.b_i = init_zero_variable(b_i,b_i_init,[1,output_channel,1,1],self.name_scope+'b_i')
            self.b_o = init_zero_variable(b_o,b_o_init,[1,output_channel,1,1],self.name_scope+'b_o')
            self.b_f = init_one_variable(b_f,b_f_init,[1,output_channel,1,1],self.name_scope+'b_f')
            self.b_g = init_zero_variable(b_g,b_g_init,[1,output_channel,1,1],self.name_scope+'b_g')
            if summ_flag:
                self.Fx_i_summ = tf.summary.histogram('Fx_i',self.Fx_i)
                self.Fx_o_summ = tf.summary.histogram('Fx_o',self.Fx_o)
                self.Fx_f_summ = tf.summary.histogram('Fx_f',self.Fx_f)
                self.Fx_g_summ = tf.summary.histogram('Fx_g',self.Fx_g)
                self.Fh_i_summ = tf.summary.histogram('Fh_i',self.Fh_i)
                self.Fh_o_summ = tf.summary.histogram('Fh_o',self.Fh_o)
                self.Fh_f_summ = tf.summary.histogram('Fh_f',self.Fh_f)
                self.Fh_g_summ = tf.summary.histogram('Fh_g',self.Fh_g)
                self.b_i_summ = tf.summary.histogram('b_i',self.b_i)
                self.b_o_summ = tf.summary.histogram('b_o',self.b_o)
                self.b_f_summ = tf.summary.histogram('b_f',self.b_f)
                self.b_g_summ = tf.summary.histogram('b_g',self.b_g)
                self.Fx_i_image = tf.summary.image('Fx_i_image',tf.transpose(self.Fx_i,[0,2,3,1]),max_outputs=50)
                self.Fx_o_image = tf.summary.image('Fx_o_image',tf.transpose(self.Fx_o,[0,2,3,1]),max_outputs=50)
                self.Fx_g_image = tf.summary.image('Fx_g_image',tf.transpose(self.Fx_g,[0,2,3,1]),max_outputs=50)
                self.Fx_f_image = tf.summary.image('Fx_f_image',tf.transpose(self.Fx_f,[0,2,3,1]),max_outputs=50)
                self.Fh_i_image = tf.summary.image('Fh_i_image',tf.transpose(self.Fh_i,[0,2,3,1]),max_outputs=50)
                self.Fh_o_image = tf.summary.image('Fh_o_image',tf.transpose(self.Fh_o,[0,2,3,1]),max_outputs=50)
                self.Fh_g_image = tf.summary.image('Fh_g_image',tf.transpose(self.Fh_g,[0,2,3,1]),max_outputs=50)
                self.Fh_f_image = tf.summary.image('Fh_f_image',tf.transpose(self.Fh_f,[0,2,3,1]),max_outputs=50)
                self.b_i_image = tf.summary.image('b_i_image',tf.transpose(self.b_i,[0,2,1,3]),max_outputs=50)
                self.b_o_image = tf.summary.image('b_o_image',tf.transpose(self.b_o,[0,2,1,3]),max_outputs=50)
                self.b_g_image = tf.summary.image('b_g_image',tf.transpose(self.b_g,[0,2,1,3]),max_outputs=50)
                self.b_f_image = tf.summary.image('b_f_image',tf.transpose(self.b_f,[0,2,1,3]),max_outputs=50)

    def __del__(self):
        del self.input_size
        del self.input_channel
        del self.output_channel
        del self.name_scope
        del self.Fx_i
        del self.Fx_o
        del self.Fx_f
        del self.Fx_g
        del self.Fh_i
        del self.Fh_o
        del self.Fh_f
        del self.Fh_g
        del self.b_i
        del self.b_o
        del self.b_f
        del self.b_g
        if self.summ_flag:
            del self.Fx_i_summ
            del self.Fx_o_summ
            del self.Fx_f_summ
            del self.Fx_g_summ
            del self.Fh_i_summ
            del self.Fh_o_summ
            del self.Fh_f_summ
            del self.Fh_g_summ
            del self.b_i_summ
            del self.b_o_summ
            del self.b_f_summ
            del self.b_g_summ
        del self.summ_flag

    def __call__(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        return self.n_steps(input_tensor,previous_hidden_state,previous_cell_state)

    def one_step(self,input_tensor,previous_hidden_state,previous_cell_state):
        '''
        input_tensor is numpy.ndarray of shape [batch_size,input_channel,input_size,1]
        previous_hidden_state is numpy.ndarray of shape [batch_size,output_channel,input_size,1]
        previous_cell_state is numpy.ndarray of shape [batch_size,output_channel,input_size,1]
        return (hidden_state,cell_state)
        '''
        with tf.name_scope(self.name_scope):
            with tf.name_scope('one_step'):
                i = tf.sigmoid(tf.add(self.b_i,tf.add(tf.nn.conv2d(input_tensor,self.Fx_i,[1,1,1,1],'SAME',data_format='NCHW'),tf.nn.conv2d(previous_hidden_state,self.Fh_i,[1,1,1,1],'SAME',data_format='NCHW'))),name='i')
                f = tf.sigmoid(tf.add(self.b_f,tf.add(tf.nn.conv2d(input_tensor,self.Fx_f,[1,1,1,1],'SAME',data_format='NCHW'),tf.nn.conv2d(previous_hidden_state,self.Fh_f,[1,1,1,1],'SAME',data_format='NCHW'))),name='f')
                o = tf.sigmoid(tf.add(self.b_o,tf.add(tf.nn.conv2d(input_tensor,self.Fx_o,[1,1,1,1],'SAME',data_format='NCHW'),tf.nn.conv2d(previous_hidden_state,self.Fh_o,[1,1,1,1],'SAME',data_format='NCHW'))),name='o')
                g = tf.tanh(tf.add(self.b_g,tf.add(tf.nn.conv2d(input_tensor,self.Fx_g,[1,1,1,1],'SAME',data_format='NCHW'),tf.nn.conv2d(previous_hidden_state,self.Fh_g,[1,1,1,1],'SAME',data_format='NCHW'))),name='g')

                cell_state = tf.add(tf.multiply(previous_cell_state,f),tf.multiply(g,i))
                hidden_state = tf.multiply(tf.tanh(cell_state),o)
                return (hidden_state,cell_state)

    def n_steps(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        '''
        input_tensor is numpy.ndarray of shape [batch_size,time_len,input_channel,input_size,1]
        previous_hidden_state is numpy.ndarray of shape [batch_size,output_channel,input_size,1]
        previous_cell_state is numpy.ndarray of shape [batch_size,output_channel,input_size,1]
        return previous_hidden_states is numpy.ndarray of shape [batch_size,time_len,output_channel,input_size,1]
        '''
        with tf.name_scope(self.name_scope):
            input_shape = input_tensor.shape.as_list()
            if previous_cell_state is None:
                previous_cell_state = tf.zeros_like(tf.nn.conv2d(input_tensor[:,0],self.Fx_i,[1,1,1,1],'SAME',data_format='NCHW'),dtype=tf.float32,name='init_cell_state')
            if previous_hidden_state is None:
                previous_hidden_state = tf.zeros_like(previous_cell_state,dtype=tf.float32,name='init_hidden_state')

            hidden_state_list = []
            for i in range(input_shape[1]):
                previous_hidden_state,previous_cell_state = self.one_step(input_tensor[:,i],previous_hidden_state,previous_cell_state)
                hidden_state_list.append(previous_hidden_state)
            hidden_state = tf.stack(hidden_state_list,axis=1,name='multiple_hidden_states')
            self.hidden_state,self.cell_state = (previous_hidden_state,previous_cell_state)
            return hidden_state

    def get_l2_loss(self):
        loss = tf.reduce_mean(tf.square(self.Fx_i))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_o))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_f))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_g))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_i))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_o))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_f))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_g))
        return loss

class AtrousConvLSTMCell:
    '''
    The class for the LSTM layer, where we replace traditional dense layer with 1D atrous convolution
    '''
    def __init__(self,input_size,input_channel,output_channel,kernel_size,rate,name_scope='AtoursConvLSTMCell',summ_flag=True,
                 Fx_i=None,Fx_i_init=None,Fx_o=None,Fx_o_init=None,Fx_f=None,Fx_f_init=None,Fx_g=None,Fx_g_init=None,
                 Fh_i=None,Fh_i_init=None,Fh_o=None,Fh_o_init=None,Fh_f=None,Fh_f_init=None,Fh_g=None,Fh_g_init=None,
                 b_i=None,b_i_init=None,b_o=None,b_o_init=None,b_f=None,b_f_init=None,b_g=None,b_g_init=None):
        self.input_size = input_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.rate = rate
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            self.Fx_i = init_ortho_random_variable(Fx_i,Fx_i_init,[kernel_size,1,input_channel,output_channel],1./((input_channel+1)*kernel_size),self.name_scope+'Fx_i')
            self.Fx_o = init_ortho_random_variable(Fx_o,Fx_o_init,[kernel_size,1,input_channel,output_channel],1./((input_channel+1)*kernel_size),self.name_scope+'Fx_o')
            self.Fx_f = init_ortho_random_variable(Fx_f,Fx_f_init,[kernel_size,1,input_channel,output_channel],1./((input_channel+1)*kernel_size),self.name_scope+'Fx_f')
            self.Fx_g = init_ortho_random_variable(Fx_g,Fx_g_init,[kernel_size,1,input_channel,output_channel],1./((input_channel+1)*kernel_size),self.name_scope+'Fx_g')
            self.Fh_i = init_ortho_random_variable(Fh_i,Fh_i_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_i')
            self.Fh_o = init_ortho_random_variable(Fh_o,Fh_o_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_o')
            self.Fh_f = init_ortho_random_variable(Fh_f,Fh_f_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_f')
            self.Fh_g = init_ortho_random_variable(Fh_g,Fh_g_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_g')
            self.b_i = init_zero_variable(b_i,b_i_init,[1,1,1,output_channel],self.name_scope+'b_i')
            self.b_o = init_zero_variable(b_o,b_o_init,[1,1,1,output_channel],self.name_scope+'b_o')
            self.b_f = init_one_variable(b_f,b_f_init,[1,1,1,output_channel],self.name_scope+'b_f')
            self.b_g = init_zero_variable(b_g,b_g_init,[1,1,1,output_channel],self.name_scope+'b_g')
            if summ_flag:
                self.Fx_i_summ = tf.summary.histogram('Fx_i',self.Fx_i)
                self.Fx_o_summ = tf.summary.histogram('Fx_o',self.Fx_o)
                self.Fx_f_summ = tf.summary.histogram('Fx_f',self.Fx_f)
                self.Fx_g_summ = tf.summary.histogram('Fx_g',self.Fx_g)
                self.Fh_i_summ = tf.summary.histogram('Fh_i',self.Fh_i)
                self.Fh_o_summ = tf.summary.histogram('Fh_o',self.Fh_o)
                self.Fh_f_summ = tf.summary.histogram('Fh_f',self.Fh_f)
                self.Fh_g_summ = tf.summary.histogram('Fh_g',self.Fh_g)
                self.b_i_summ = tf.summary.histogram('b_i',self.b_i)
                self.b_o_summ = tf.summary.histogram('b_o',self.b_o)
                self.b_f_summ = tf.summary.histogram('b_f',self.b_f)
                self.b_g_summ = tf.summary.histogram('b_g',self.b_g)
                self.Fx_i_image = tf.summary.image('Fx_i_image',tf.transpose(self.Fx_i,[0,2,3,1]),max_outputs=50)
                self.Fx_o_image = tf.summary.image('Fx_o_image',tf.transpose(self.Fx_o,[0,2,3,1]),max_outputs=50)
                self.Fx_g_image = tf.summary.image('Fx_g_image',tf.transpose(self.Fx_g,[0,2,3,1]),max_outputs=50)
                self.Fx_f_image = tf.summary.image('Fx_f_image',tf.transpose(self.Fx_f,[0,2,3,1]),max_outputs=50)
                self.Fh_i_image = tf.summary.image('Fh_i_image',tf.transpose(self.Fh_i,[0,2,3,1]),max_outputs=50)
                self.Fh_o_image = tf.summary.image('Fh_o_image',tf.transpose(self.Fh_o,[0,2,3,1]),max_outputs=50)
                self.Fh_g_image = tf.summary.image('Fh_g_image',tf.transpose(self.Fh_g,[0,2,3,1]),max_outputs=50)
                self.Fh_f_image = tf.summary.image('Fh_f_image',tf.transpose(self.Fh_f,[0,2,3,1]),max_outputs=50)
                self.b_i_image = tf.summary.image('b_i_image',tf.transpose(self.b_i,[0,1,3,2]),max_outputs=50)
                self.b_o_image = tf.summary.image('b_o_image',tf.transpose(self.b_o,[0,1,3,2]),max_outputs=50)
                self.b_g_image = tf.summary.image('b_g_image',tf.transpose(self.b_g,[0,1,3,2]),max_outputs=50)
                self.b_f_image = tf.summary.image('b_f_image',tf.transpose(self.b_f,[0,1,3,2]),max_outputs=50)

    def __del__(self):
        del self.input_size
        del self.input_channel
        del self.output_channel
        del self.name_scope
        del self.Fx_i
        del self.Fx_o
        del self.Fx_f
        del self.Fx_g
        del self.Fh_i
        del self.Fh_o
        del self.Fh_f
        del self.Fh_g
        del self.b_i
        del self.b_o
        del self.b_f
        del self.b_g
        if self.summ_flag:
            del self.Fx_i_summ
            del self.Fx_o_summ
            del self.Fx_f_summ
            del self.Fx_g_summ
            del self.Fh_i_summ
            del self.Fh_o_summ
            del self.Fh_f_summ
            del self.Fh_g_summ
            del self.b_i_summ
            del self.b_o_summ
            del self.b_f_summ
            del self.b_g_summ
        del self.summ_flag

    def __call__(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        return self.n_steps(input_tensor,previous_hidden_state,previous_cell_state)

    def one_step(self,input_tensor,previous_hidden_state,previous_cell_state):
        '''
        input_tensor is numpy.ndarray of shape [batch_size,input_size,1,input_channel]
        previous_hidden_state is numpy.ndarray of shape [batch_size,input_size,1,output_channel]
        previous_cell_state is numpy.ndarray of shape [batch_size,input_size,1,output_channel]
        return (hidden_state,cell_state)
        '''
        with tf.name_scope(self.name_scope):
            with tf.name_scope('one_step'):
                i = tf.sigmoid(tf.add(self.b_i,tf.add(tf.nn.atrous_conv2d(input_tensor,self.Fx_i,self.rate,'SAME'),tf.nn.atrous_conv2d(previous_hidden_state,self.Fh_i,self.rate,'SAME',))),name='i')
                f = tf.sigmoid(tf.add(self.b_f,tf.add(tf.nn.atrous_conv2d(input_tensor,self.Fx_f,self.rate,'SAME',),tf.nn.atrous_conv2d(previous_hidden_state,self.Fh_f,self.rate,'SAME',))),name='f')
                o = tf.sigmoid(tf.add(self.b_o,tf.add(tf.nn.atrous_conv2d(input_tensor,self.Fx_o,self.rate,'SAME',),tf.nn.atrous_conv2d(previous_hidden_state,self.Fh_o,self.rate,'SAME',))),name='o')
                g = tf.tanh(tf.add(self.b_g,tf.add(tf.nn.atrous_conv2d(input_tensor,self.Fx_g,self.rate,'SAME'),tf.nn.atrous_conv2d(previous_hidden_state,self.Fh_g,self.rate,'SAME'))),name='g')

                cell_state = tf.add(tf.multiply(previous_cell_state,f),tf.multiply(g,i))
                hidden_state = tf.multiply(tf.tanh(cell_state),o)
                return (hidden_state,cell_state)

    def n_steps(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        '''
        input_tensor is numpy.ndarray of shape [batch_size,input_size,1,input_channel]
        previous_hidden_state is numpy.ndarray of shape [batch_size,input_size,1,output_channel]
        previous_cell_state is numpy.ndarray of shape [batch_size,input_size,1,output_channel]
        return previous_hidden_states is numpy.ndarray of shape [batch_size,time_len,output_channel,input_size,1]
        '''
        with tf.name_scope(self.name_scope):
            input_shape = input_tensor.shape.as_list()
            if previous_cell_state is None:
                previous_cell_state = tf.zeros_like(tf.nn.conv2d(input_tensor[:,0],self.Fx_i,'SAME'),dtype=tf.float32,name='init_cell_state')
            if previous_hidden_state is None:
                previous_hidden_state = tf.zeros_like(previous_cell_state,dtype=tf.float32,name='init_hidden_state')

            hidden_state_list = []
            for i in range(input_shape[1]):
                previous_hidden_state,previous_cell_state = self.one_step(input_tensor[:,i],previous_hidden_state,previous_cell_state)
                hidden_state_list.append(previous_hidden_state)
            hidden_state = tf.stack(hidden_state_list,axis=1,name='multiple_hidden_states')
            self.hidden_state,self.cell_state = (previous_hidden_state,previous_cell_state)
            return hidden_state

    def get_l2_loss(self):
        loss = tf.reduce_mean(tf.square(self.Fx_i))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_o))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_f))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_g))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_i))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_o))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_f))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_g))
        return loss

class SAttenAtrousConvLSTMCell:
    '''
    The class for the LSTM layer with self-attention, where we replace traditional dense layer with 1D atrous convolution
    '''
    def __init__(self,input_size,input_channel,output_channel,kernel_size,rate,name_scope='SAttenAtrousConvLSTMCell',summ_flag=True,
                 Fx_i=None,Fx_i_init=None,Fx_o=None,Fx_o_init=None,Fx_f=None,Fx_f_init=None,Fx_g=None,Fx_g_init=None,
                 Fh_i=None,Fh_i_init=None,Fh_o=None,Fh_o_init=None,Fh_f=None,Fh_f_init=None,Fh_g=None,Fh_g_init=None,
                 b_i=None,b_i_init=None,b_o=None,b_o_init=None,b_f=None,b_f_init=None,b_g=None,b_g_init=None):
        self.input_size = input_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.rate = rate
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            self.Fx_i = init_ortho_random_variable(Fx_i,Fx_i_init,[kernel_size,1,output_channel+input_channel,output_channel],1./((output_channel+input_channel+1)*kernel_size),self.name_scope+'Fx_i')
            self.Fx_o = init_ortho_random_variable(Fx_o,Fx_o_init,[kernel_size,1,output_channel+input_channel,output_channel],1./((output_channel+input_channel+1)*kernel_size),self.name_scope+'Fx_o')
            self.Fx_f = init_ortho_random_variable(Fx_f,Fx_f_init,[kernel_size,1,output_channel+input_channel,output_channel],1./((output_channel+input_channel+1)*kernel_size),self.name_scope+'Fx_f')
            self.Fx_g = init_ortho_random_variable(Fx_g,Fx_g_init,[kernel_size,1,output_channel+input_channel,output_channel],1./((output_channel+input_channel+1)*kernel_size),self.name_scope+'Fx_g')
            self.Fh_i = init_ortho_random_variable(Fh_i,Fh_i_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_i')
            self.Fh_o = init_ortho_random_variable(Fh_o,Fh_o_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_o')
            self.Fh_f = init_ortho_random_variable(Fh_f,Fh_f_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_f')
            self.Fh_g = init_ortho_random_variable(Fh_g,Fh_g_init,[kernel_size,1,output_channel,output_channel],1./((output_channel+1)*kernel_size),self.name_scope+'Fh_g')
            self.b_i = init_zero_variable(b_i,b_i_init,[1,1,1,output_channel],self.name_scope+'b_i')
            self.b_o = init_zero_variable(b_o,b_o_init,[1,1,1,output_channel],self.name_scope+'b_o')
            self.b_f = init_one_variable(b_f,b_f_init,[1,1,1,output_channel],self.name_scope+'b_f')
            self.b_g = init_zero_variable(b_g,b_g_init,[1,1,1,output_channel],self.name_scope+'b_g')
            if summ_flag:
                self.Fx_i_summ = tf.summary.histogram('Fx_i',self.Fx_i)
                self.Fx_o_summ = tf.summary.histogram('Fx_o',self.Fx_o)
                self.Fx_f_summ = tf.summary.histogram('Fx_f',self.Fx_f)
                self.Fx_g_summ = tf.summary.histogram('Fx_g',self.Fx_g)
                self.Fh_i_summ = tf.summary.histogram('Fh_i',self.Fh_i)
                self.Fh_o_summ = tf.summary.histogram('Fh_o',self.Fh_o)
                self.Fh_f_summ = tf.summary.histogram('Fh_f',self.Fh_f)
                self.Fh_g_summ = tf.summary.histogram('Fh_g',self.Fh_g)
                self.b_i_summ = tf.summary.histogram('b_i',self.b_i)
                self.b_o_summ = tf.summary.histogram('b_o',self.b_o)
                self.b_f_summ = tf.summary.histogram('b_f',self.b_f)
                self.b_g_summ = tf.summary.histogram('b_g',self.b_g)
                self.Fx_i_image = tf.summary.image('Fx_i_image',tf.transpose(self.Fx_i,[0,2,3,1]),max_outputs=50)
                self.Fx_o_image = tf.summary.image('Fx_o_image',tf.transpose(self.Fx_o,[0,2,3,1]),max_outputs=50)
                self.Fx_g_image = tf.summary.image('Fx_g_image',tf.transpose(self.Fx_g,[0,2,3,1]),max_outputs=50)
                self.Fx_f_image = tf.summary.image('Fx_f_image',tf.transpose(self.Fx_f,[0,2,3,1]),max_outputs=50)
                self.Fh_i_image = tf.summary.image('Fh_i_image',tf.transpose(self.Fh_i,[0,2,3,1]),max_outputs=50)
                self.Fh_o_image = tf.summary.image('Fh_o_image',tf.transpose(self.Fh_o,[0,2,3,1]),max_outputs=50)
                self.Fh_g_image = tf.summary.image('Fh_g_image',tf.transpose(self.Fh_g,[0,2,3,1]),max_outputs=50)
                self.Fh_f_image = tf.summary.image('Fh_f_image',tf.transpose(self.Fh_f,[0,2,3,1]),max_outputs=50)
                self.b_i_image = tf.summary.image('b_i_image',tf.transpose(self.b_i,[0,1,3,2]),max_outputs=50)
                self.b_o_image = tf.summary.image('b_o_image',tf.transpose(self.b_o,[0,1,3,2]),max_outputs=50)
                self.b_g_image = tf.summary.image('b_g_image',tf.transpose(self.b_g,[0,1,3,2]),max_outputs=50)
                self.b_f_image = tf.summary.image('b_f_image',tf.transpose(self.b_f,[0,1,3,2]),max_outputs=50)

    def __del__(self):
        del self.input_size
        del self.input_channel
        del self.output_channel
        del self.name_scope
        del self.Fx_i
        del self.Fx_o
        del self.Fx_f
        del self.Fx_g
        del self.Fh_i
        del self.Fh_o
        del self.Fh_f
        del self.Fh_g
        del self.b_i
        del self.b_o
        del self.b_f
        del self.b_g
        if self.summ_flag:
            del self.Fx_i_summ
            del self.Fx_o_summ
            del self.Fx_f_summ
            del self.Fx_g_summ
            del self.Fh_i_summ
            del self.Fh_o_summ
            del self.Fh_f_summ
            del self.Fh_g_summ
            del self.b_i_summ
            del self.b_o_summ
            del self.b_f_summ
            del self.b_g_summ
        del self.summ_flag

    def __call__(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        return self.n_steps(input_tensor,previous_hidden_state,previous_cell_state)

    def one_step(self,input_tensor,previous_hidden_state,previous_cell_state):
        '''
        input_tensor is numpy.ndarray of shape [batch_size,input_size,1,output_channel+input_channel]
        previous_hidden_state is numpy.ndarray of shape [batch_size,input_size,1,output_channel]
        previous_cell_state is numpy.ndarray of shape [batch_size,input_size,1,output_channel]
        return (hidden_state,cell_state)
        '''
        with tf.name_scope(self.name_scope):
            with tf.name_scope('one_step'):
                i = tf.sigmoid(tf.add(self.b_i,tf.add(tf.nn.atrous_conv2d(input_tensor,self.Fx_i,self.rate,'SAME'),tf.nn.atrous_conv2d(previous_hidden_state,self.Fh_i,self.rate,'SAME',))),name='i')
                f = tf.sigmoid(tf.add(self.b_f,tf.add(tf.nn.atrous_conv2d(input_tensor,self.Fx_f,self.rate,'SAME',),tf.nn.atrous_conv2d(previous_hidden_state,self.Fh_f,self.rate,'SAME',))),name='f')
                o = tf.sigmoid(tf.add(self.b_o,tf.add(tf.nn.atrous_conv2d(input_tensor,self.Fx_o,self.rate,'SAME',),tf.nn.atrous_conv2d(previous_hidden_state,self.Fh_o,self.rate,'SAME',))),name='o')
                g = tf.tanh(tf.add(self.b_g,tf.add(tf.nn.atrous_conv2d(input_tensor,self.Fx_g,self.rate,'SAME'),tf.nn.atrous_conv2d(previous_hidden_state,self.Fh_g,self.rate,'SAME'))),name='g')

                cell_state = tf.add(tf.multiply(previous_cell_state,f),tf.multiply(g,i))
                hidden_state = tf.multiply(tf.tanh(cell_state),o)
                return (hidden_state,cell_state)

    def n_steps(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        '''
        input_tensor is numpy.ndarray of shape [batch_size,input_size,1,input_channel]
        previous_hidden_state is numpy.ndarray of shape [batch_size,input_size,1,output_channel]
        previous_cell_state is numpy.ndarray of shape [batch_size,input_size,1,output_channel]
        return previous_hidden_states is numpy.ndarray of shape [batch_size,time_len,output_channel,input_size,1]
        '''
        with tf.name_scope(self.name_scope):
            input_shape = input_tensor.shape.as_list()
            if previous_cell_state is None:
                previous_cell_state = tf.zeros_like(tf.nn.atrous_conv2d(input_tensor[:,0],tf.ones([self.kernel_size,1,self.input_channel,self.output_channel]),self.rate,'SAME'),dtype=tf.float32,name='init_cell_state')
            if previous_hidden_state is None:
                previous_hidden_state = tf.zeros_like(previous_cell_state,dtype=tf.float32,name='init_hidden_state')

            hidden_state_list = []
            for i in range(input_shape[1]):
                with tf.name_scope('Self_Attention'):
                    if i > 0:
                        logits = tf.reduce_sum(tf.expand_dims(previous_hidden_state,axis=1)*tf.stack(hidden_state_list,axis=1),axis=(2,3,4),keep_dims=True,name='logits')
                        weight = tf.nn.softmax(logits,dim=1,name='weight')
                        atten_value = tf.reduce_sum(weight*tf.stack(hidden_state_list,axis=1),axis=1)
                    else:
                        atten_value = tf.zeros_like(previous_hidden_state)
                previous_hidden_state,previous_cell_state = self.one_step(tf.concat([input_tensor[:,i],atten_value],axis=-1),previous_hidden_state,previous_cell_state)
                hidden_state_list.append(previous_hidden_state)
            hidden_state = tf.stack(hidden_state_list,axis=1,name='multiple_hidden_states')
            self.hidden_state,self.cell_state = (previous_hidden_state,previous_cell_state)
            return hidden_state

    def get_l2_loss(self):
        loss = tf.reduce_mean(tf.square(self.Fx_i))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_o))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_f))
        loss = loss + tf.reduce_mean(tf.square(self.Fx_g))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_i))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_o))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_f))
        loss = loss + tf.reduce_mean(tf.square(self.Fh_g))
        return loss

def ResAtrousConvLSTMlayer(D,T,input_channel,middle_channel,output_channel,kernel_size,rate1,rate2,sample_size,input_tensor,reset_flag,activation_func=tf.nn.relu,name_scope='OneResidualLayer',summ_flag=False):
    '''
    input_tensor has shape [?,T,D,1,input_channel,] of type tf.float32
    reset_flag is a tensor with shape [] of type tf.bool
    input_flag is a tensor with shape [] of type tf.bool
    previous_hidden_state1 has shape [?,T,D,1,output_channel] of type tf.float32
    previous_cell_state1 has shape [?,T,D,1,output_channel] of type tf.float32
    clstm_out1 has shape [?,T,D,1,output_channel]
    clstm is of type AtrousConvLSTMCell
    conv is of type ConvCell
    '''
    with tf.name_scope(name_scope):
#        with tf.name_scope('Input'):
#            previous_hidden_state1 = tf.placeholder(tf.float32,[None,D,1,middle_channel],name='previous_hidden_state1')
#            previous_cell_state1 = tf.placeholder(tf.float32,[None,D,1,middle_channel],name='previous_cell_state1')
#            previous_hidden_state2 = tf.placeholder(tf.float32,[None,D,1,output_channel],name='previous_hidden_state2')
#            previous_cell_state2 = tf.placeholder(tf.float32,[None,D,1,output_channel],name='previous_cell_state2')

        with tf.name_scope('Memory_hidden_states'):
            previous_hidden_state1_mem = tf.get_variable(name_scope+'/previous_hidden_state1_mem',[sample_size,D,1,middle_channel],dtype=tf.float32,initializer=tf.zeros_initializer())
            previous_cell_state1_mem = tf.get_variable(name_scope+'/previous_cell_state1_mem',[sample_size,D,1,middle_channel],dtype=tf.float32,initializer=tf.zeros_initializer())
            previous_hidden_state2_mem = tf.get_variable(name_scope+'/previous_hidden_state2_mem',[sample_size,D,1,output_channel],dtype=tf.float32,initializer=tf.zeros_initializer())
            previous_cell_state2_mem = tf.get_variable(name_scope+'/previous_cell_state2_mem',[sample_size,D,1,output_channel],dtype=tf.float32,initializer=tf.zeros_initializer())

        previous_hidden_state1 = tf.cond(reset_flag,lambda: previous_hidden_state1_mem.assign(tf.zeros(previous_hidden_state1_mem.shape)),lambda: previous_hidden_state1_mem)
        previous_cell_state1 = tf.cond(reset_flag,lambda: previous_cell_state1_mem.assign(tf.zeros(previous_cell_state1_mem.shape)),lambda: previous_cell_state1_mem)
        previous_hidden_state2 = tf.cond(reset_flag,lambda: previous_hidden_state2_mem.assign(tf.zeros(previous_hidden_state2_mem.shape)),lambda: previous_hidden_state2_mem)
        previous_cell_state2 = tf.cond(reset_flag,lambda: previous_cell_state2_mem.assign(tf.zeros(previous_cell_state2_mem.shape)),lambda: previous_cell_state2_mem)

        with tf.name_scope('Conv_Layer'):
            print(name_scope,'Conv Layer')
            conv = ConvCell([1,1,input_channel,output_channel],data_format='NHWC',name_scope='Conv')
            if input_channel == output_channel:
                conv_out = input_tensor
            else:
                conv_out = tf.reshape(conv(tf.reshape(input_tensor,[-1,D,1,input_channel])),[-1,T,D,1,output_channel],name='conv_out')
            print(conv_out.shape)
            if summ_flag:
                tf.summary.histogram('conv_out',conv_out)
                tf.summary.image('conv_out_image',tf.transpose(conv_out[0],perm=[0,1,3,2]),max_outputs=7)  #size=T-1, W*H=D*param.output_channel

        with tf.name_scope('AtrousConvLSTM_Layer'):
            print('AtrousConvLSTM Layer')
            clstm1 = AtrousConvLSTMCell(D,input_channel,middle_channel,kernel_size,rate1,'ConvLSTM1',summ_flag)
            clstm2 = AtrousConvLSTMCell(D,middle_channel,output_channel,kernel_size,rate2,'ConvLSTM2',summ_flag)
            clstm_out1 = activation_func(clstm1(input_tensor,previous_hidden_state=previous_hidden_state1,previous_cell_state=previous_cell_state1),name='clstm_out1')
            clstm_out2 = clstm2(clstm_out1,previous_hidden_state=previous_hidden_state2,previous_cell_state=previous_cell_state2)
            clstm_out = tf.add(clstm_out2,conv_out,name='clstm_out')
            print(clstm_out.shape)
            if summ_flag:
                tf.summary.histogram('clstm_out1',clstm_out1)
                tf.summary.image('clstm_out1_image',tf.transpose(clstm_out1[0],perm=[0,1,3,2]),max_outputs=7)  #size=T-1, W*H=D*middle_channel
                tf.summary.histogram('clstm_out2',clstm_out2)
                tf.summary.image('clstm_out2_image',tf.transpose(clstm_out2[0],perm=[0,1,3,2]),max_outputs=7)  #size=T-1, W*H=D*output_channel

            update_states = [previous_hidden_state1_mem.assign(clstm1.hidden_state).op,\
                             previous_cell_state1_mem.assign(clstm1.cell_state).op,\
                             previous_hidden_state2_mem.assign(clstm2.hidden_state).op,\
                             previous_cell_state2_mem.assign(clstm2.cell_state).op,]
        return clstm_out,update_states,conv,clstm1,clstm2

def ResSAttenAtrousConvLSTMlayer(D,T,input_channel,middle_channel,output_channel,kernel_size,rate1,rate2,sample_size,input_tensor,reset_flag,activation_func=tf.nn.relu,name_scope='OneResidualLayer',summ_flag=False):
    '''
    input_tensor has shape [?,T,D,1,input_channel,] of type tf.float32
    reset_flag is a tensor with shape [] of type tf.bool
    input_flag is a tensor with shape [] of type tf.bool
    previous_hidden_state1 has shape [?,T,D,1,output_channel] of type tf.float32
    previous_cell_state1 has shape [?,T,D,1,output_channel] of type tf.float32
    clstm_out1 has shape [?,T,D,1,output_channel]
    clstm is of type AtrousConvLSTMCell
    conv is of type ConvCell
    '''
    with tf.name_scope(name_scope):
#        with tf.name_scope('Input'):
#            previous_hidden_state1 = tf.placeholder(tf.float32,[None,D,1,middle_channel],name='previous_hidden_state1')
#            previous_cell_state1 = tf.placeholder(tf.float32,[None,D,1,middle_channel],name='previous_cell_state1')
#            previous_hidden_state2 = tf.placeholder(tf.float32,[None,D,1,output_channel],name='previous_hidden_state2')
#            previous_cell_state2 = tf.placeholder(tf.float32,[None,D,1,output_channel],name='previous_cell_state2')

        with tf.name_scope('Memory_hidden_states'):
            previous_hidden_state1_mem = tf.get_variable(name_scope+'/previous_hidden_state1_mem',[sample_size,D,1,middle_channel],dtype=tf.float32,initializer=tf.zeros_initializer())
            previous_cell_state1_mem = tf.get_variable(name_scope+'/previous_cell_state1_mem',[sample_size,D,1,middle_channel],dtype=tf.float32,initializer=tf.zeros_initializer())
            previous_hidden_state2_mem = tf.get_variable(name_scope+'/previous_hidden_state2_mem',[sample_size,D,1,output_channel],dtype=tf.float32,initializer=tf.zeros_initializer())
            previous_cell_state2_mem = tf.get_variable(name_scope+'/previous_cell_state2_mem',[sample_size,D,1,output_channel],dtype=tf.float32,initializer=tf.zeros_initializer())

        previous_hidden_state1 = tf.cond(reset_flag,lambda: previous_hidden_state1_mem.assign(tf.zeros(previous_hidden_state1_mem.shape)),lambda: previous_hidden_state1_mem)
        previous_cell_state1 = tf.cond(reset_flag,lambda: previous_cell_state1_mem.assign(tf.zeros(previous_cell_state1_mem.shape)),lambda: previous_cell_state1_mem)
        previous_hidden_state2 = tf.cond(reset_flag,lambda: previous_hidden_state2_mem.assign(tf.zeros(previous_hidden_state2_mem.shape)),lambda: previous_hidden_state2_mem)
        previous_cell_state2 = tf.cond(reset_flag,lambda: previous_cell_state2_mem.assign(tf.zeros(previous_cell_state2_mem.shape)),lambda: previous_cell_state2_mem)

        with tf.name_scope('Conv_Layer'):
            print(name_scope,'Conv Layer')
            conv = ConvCell([1,1,input_channel,output_channel],data_format='NHWC',name_scope='Conv')
            if input_channel == output_channel:
                conv_out = input_tensor
            else:
                conv_out = tf.reshape(conv(tf.reshape(input_tensor,[-1,D,1,input_channel])),[-1,T,D,1,output_channel],name='conv_out')
            print(conv_out.shape)
            if summ_flag:
                tf.summary.histogram('conv_out',conv_out)
                tf.summary.image('conv_out_image',tf.transpose(conv_out[0],perm=[0,1,3,2]),max_outputs=7)  #size=T-1, W*H=D*param.output_channel

        with tf.name_scope('AtrousConvLSTM_Layer'):
            print('SAttenAtrousConvLSTM Layer')
            clstm1 = SAttenAtrousConvLSTMCell(D,input_channel,middle_channel,kernel_size,rate1,'ConvLSTM1',summ_flag)
            clstm2 = SAttenAtrousConvLSTMCell(D,middle_channel,output_channel,kernel_size,rate2,'ConvLSTM2',summ_flag)
            clstm_out1 = activation_func(clstm1(input_tensor,previous_hidden_state=previous_hidden_state1,previous_cell_state=previous_cell_state1),name='clstm_out1')
            clstm_out2 = clstm2(clstm_out1,previous_hidden_state=previous_hidden_state2,previous_cell_state=previous_cell_state2)
            clstm_out = tf.add(clstm_out2,conv_out,name='clstm_out')
            print(clstm_out.shape)
            if summ_flag:
                tf.summary.histogram('clstm_out1',clstm_out1)
                tf.summary.image('clstm_out1_image',tf.transpose(clstm_out1[0],perm=[0,1,3,2]),max_outputs=7)  #size=T-1, W*H=D*middle_channel
                tf.summary.histogram('clstm_out2',clstm_out2)
                tf.summary.image('clstm_out2_image',tf.transpose(clstm_out2[0],perm=[0,1,3,2]),max_outputs=7)  #size=T-1, W*H=D*output_channel

            update_states = [previous_hidden_state1_mem.assign(clstm1.hidden_state).op,\
                             previous_cell_state1_mem.assign(clstm1.cell_state).op,\
                             previous_hidden_state2_mem.assign(clstm2.hidden_state).op,\
                             previous_cell_state2_mem.assign(clstm2.cell_state).op,]
        return clstm_out,update_states,conv,clstm1,clstm2

def ResConvLSTMlayer(D,T,input_channel,middle_channel,output_channel,kernel_size,input_tensor,activation_func=tf.nn.relu,name_scope='OneResidualLayer',summ_flag=False):
    '''
    input_tensor has shape [?,T,input_channel,D,1] of type tf.float32
    previous_hidden_state1 has shape [?,T,output_channel,D,1] of type tf.float32
    previous_cell_state1 has shape [?,T,output_channel,D,1] of type tf.float32
    clstm_out1 has shape [?,T,output_channel,D,1]
    clstm is of type ConvLSTMCell
    conv is of type ConvCell
    '''
    with tf.name_scope(name_scope):
        with tf.name_scope('Input'):
            previous_hidden_state1 = tf.placeholder(tf.float32,[None,middle_channel,D,1],name='previous_hidden_state1')
            previous_cell_state1 = tf.placeholder(tf.float32,[None,middle_channel,D,1],name='previous_cell_state1')
            previous_hidden_state2 = tf.placeholder(tf.float32,[None,output_channel,D,1],name='previous_hidden_state2')
            previous_cell_state2 = tf.placeholder(tf.float32,[None,output_channel,D,1],name='previous_cell_state2')

        with tf.name_scope('Conv_Layer'):
            print('Conv Layer')
            conv = ConvCell([1,1,input_channel,output_channel],name_scope='Conv')
            conv_out = tf.reshape(conv(tf.reshape(input_tensor,[-1,input_channel,D,1])),[-1,T,output_channel,D,1],name='conv_out')
            print(conv_out.shape)
            if summ_flag:
                tf.summary.histogram('conv_out',conv_out)
                tf.summary.image('conv_out_image',tf.transpose(conv_out[0],perm=[0,2,1,3]),max_outputs=7)  #size=T-1, W*H=D*param.channel_num

        with tf.name_scope('ConvLSTM_Layer'):
            print('ConvLSTM Layer')
            clstm1 = ConvLSTMCell(D,input_channel,middle_channel,kernel_size,'ConvLSTM1',summ_flag)
            clstm2 = ConvLSTMCell(D,middle_channel,output_channel,kernel_size,'ConvLSTM2',summ_flag)
            clstm_out1 = tf.nn.relu(clstm1(input_tensor,previous_hidden_state=previous_hidden_state1,previous_cell_state=previous_cell_state1),'clstm_out1')
            clstm_out2 = clstm2(clstm_out1,previous_hidden_state=previous_hidden_state2,previous_cell_state=previous_cell_state2)
            clstm_out = tf.add(clstm_out2,conv_out,name='clstm_out')
            print(clstm_out.shape)
            if summ_flag:
                tf.summary.histogram('clstm_out1',clstm_out1)
                tf.summary.image('clstm_out1_image',tf.transpose(clstm_out1[0],perm=[0,2,1,3]),max_outputs=7)  #size=T-1, W*H=D*middle_channel
                tf.summary.histogram('clstm_out2',clstm_out2)
                tf.summary.image('clstm_out2_image',tf.transpose(clstm_out2[0],perm=[0,2,1,3]),max_outputs=7)  #size=T-1, W*H=D*output_channel

        return clstm_out,[previous_hidden_state1,previous_hidden_state2],[previous_cell_state1,previous_cell_state2],conv,clstm1,clstm2
