#!/usr/bin/env python
# coding=utf-8

# 20180805 by tanghao
# reorganized from nn_parts.py
# This file contains RNN-related layers

import tensorflow as tf
from .init_var import *
from .norm import LayerNorm

class LSTMCell:
    '''
    The class for the LSTM layer, which contains the weight matrices and biases and one function to generate LSTM output
    '''
    def __init__(self,input_size,output_size,peephole_flag=True,name_scope='LSTMCell',summ_flag=True,
                 Wx_i=None,Wx_i_init=None,Wx_o=None,Wx_o_init=None,Wx_f=None,Wx_f_init=None,Wx_g=None,Wx_g_init=None,
                 Wh_i=None,Wh_i_init=None,Wh_o=None,Wh_o_init=None,Wh_f=None,Wh_f_init=None,Wh_g=None,Wh_g_init=None,
                 Wc_i=None,Wc_i_init=None,Wc_o=None,Wc_o_init=None,Wc_f=None,Wc_f_init=None,Wc_g=None,Wc_g_init=None,
                 b_i=None,b_i_init=None,b_o=None,b_o_init=None,b_f=None,b_f_init=None,b_g=None,b_g_init=None):
        self.input_size = input_size
        self.output_size = output_size
        self.summ_flag = summ_flag
        self.peephole_flag = peephole_flag
        with tf.name_scope(name_scope) as self.name_scope:
            with tf.name_scope('initialize_variables'):
                self.Wx_i = init_random_variable(Wx_i,Wx_i_init,[input_size,output_size],1./(input_size+output_size),'Wx_i')
                self.Wx_o = init_random_variable(Wx_o,Wx_o_init,[input_size,output_size],1./(input_size+output_size),'Wx_o')
                self.Wx_f = init_random_variable(Wx_f,Wx_f_init,[input_size,output_size],1./(input_size+output_size),'Wx_f')
                self.Wx_g = init_random_variable(Wx_g,Wx_g_init,[input_size,output_size],1./(input_size+output_size),'Wx_g')
                self.Wh_i = init_random_variable(Wh_i,Wh_i_init,[output_size,output_size],1./(output_size+output_size),'Wh_i')
                self.Wh_o = init_random_variable(Wh_o,Wh_o_init,[output_size,output_size],1./(output_size+output_size),'Wh_o')
                self.Wh_f = init_random_variable(Wh_f,Wh_f_init,[output_size,output_size],1./(output_size+output_size),'Wh_f')
                self.Wh_g = init_random_variable(Wh_g,Wh_g_init,[output_size,output_size],1./(output_size+output_size),'Wh_g')
                if self.peephole_flag:
                    self.Wc_i = init_random_variable(Wc_i,Wc_i_init,[output_size,output_size],1./(output_size+output_size),'Wc_i')
                    self.Wc_o = init_random_variable(Wc_o,Wc_o_init,[output_size,output_size],1./(output_size+output_size),'Wc_o')
                    self.Wc_f = init_random_variable(Wc_f,Wc_f_init,[output_size,output_size],1./(output_size+output_size),'Wc_f')
                    self.Wc_g = init_random_variable(Wc_g,Wc_g_init,[output_size,output_size],1./(output_size+output_size),'Wc_g')
                self.b_i = init_zero_variable(b_i,b_i_init,[output_size],'b_i')
                self.b_o = init_zero_variable(b_o,b_o_init,[output_size],'b_o')
                self.b_f = init_one_variable(b_f,b_f_init,[output_size],'b_f')
                self.b_g = init_zero_variable(b_g,b_g_init,[output_size],'b_g')
                if summ_flag:
                    self.Wx_i_summ = tf.summary.histogram('Wx_i',self.Wx_i)
                    self.Wx_o_summ = tf.summary.histogram('Wx_o',self.Wx_o)
                    self.Wx_f_summ = tf.summary.histogram('Wx_f',self.Wx_f)
                    self.Wx_g_summ = tf.summary.histogram('Wx_g',self.Wx_g)
                    self.Wh_i_summ = tf.summary.histogram('Wh_i',self.Wh_i)
                    self.Wh_o_summ = tf.summary.histogram('Wh_o',self.Wh_o)
                    self.Wh_f_summ = tf.summary.histogram('Wh_f',self.Wh_f)
                    self.Wh_g_summ = tf.summary.histogram('Wh_g',self.Wh_g)
                    if self.peephole_flag:
                        self.Wc_i_summ = tf.summary.histogram('Wc_i',self.Wc_i)
                        self.Wc_o_summ = tf.summary.histogram('Wc_o',self.Wc_o)
                        self.Wc_f_summ = tf.summary.histogram('Wc_f',self.Wc_f)
                        self.Wc_g_summ = tf.summary.histogram('Wc_g',self.Wc_g)
                    self.b_i_summ = tf.summary.histogram('b_i',self.b_i)
                    self.b_o_summ = tf.summary.histogram('b_o',self.b_o)
                    self.b_f_summ = tf.summary.histogram('b_f',self.b_f)
                    self.b_g_summ = tf.summary.histogram('b_g',self.b_g)

    def __del__(self):
        del self.input_size
        del self.output_size
        del self.name_scope
        del self.Wx_i
        del self.Wx_o
        del self.Wx_f
        del self.Wx_g
        del self.Wh_i
        del self.Wh_o
        del self.Wh_f
        del self.Wh_g
        if self.peephole_flag:
            del self.Wc_i
            del self.Wc_o
            del self.Wc_f
            del self.Wc_g
        del self.b_i
        del self.b_o
        del self.b_f
        del self.b_g
        if self.summ_flag:
            del self.Wx_i_summ
            del self.Wx_o_summ
            del self.Wx_f_summ
            del self.Wx_g_summ
            del self.Wh_i_summ
            del self.Wh_o_summ
            del self.Wh_f_summ
            del self.Wh_g_summ
            if self.peephole_flag:
                del self.Wc_i_summ
                del self.Wc_o_summ
                del self.Wc_f_summ
                del self.Wc_g_summ
            del self.b_i_summ
            del self.b_o_summ
            del self.b_f_summ
            del self.b_g_summ
        del self.summ_flag
        del self.peephole_flag

    def __call__(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        return self.n_steps_new(input_tensor,previous_hidden_state,previous_cell_state)

    def one_step(self,input_tensor,previous_hidden_state,previous_cell_state):
        with tf.name_scope(self.name_scope):
            with tf.name_scope('one_step'):
                li = tf.add(self.b_i,tf.add(tf.matmul(input_tensor,self.Wx_i),tf.matmul(previous_hidden_state,self.Wh_i)))
                lo = tf.add(self.b_o,tf.add(tf.matmul(input_tensor,self.Wx_o),tf.matmul(previous_hidden_state,self.Wh_o)))
                lf = tf.add(self.b_f,tf.add(tf.matmul(input_tensor,self.Wx_f),tf.matmul(previous_hidden_state,self.Wh_f)))
                lg = tf.add(self.b_g,tf.add(tf.matmul(input_tensor,self.Wx_g),tf.matmul(previous_hidden_state,self.Wh_g)))
                if self.peephole_flag:
                    li = tf.add(li,tf.matmul(previous_cell_state,self.Wc_i))
                    lo = tf.add(lo,tf.matmul(previous_cell_state,self.Wc_o))
                    lf = tf.add(lf,tf.matmul(previous_cell_state,self.Wc_f))
                    lg = tf.add(lg,tf.matmul(previous_cell_state,self.Wc_g))
                i = tf.sigmoid(li,name='i')
                o = tf.sigmoid(lo,name='o')
                f = tf.sigmoid(lf,name='f')
                cell_state = tf.add(tf.multiply(previous_cell_state,f),tf.multiply(g,i),name='cell_state')
                hidden_state = tf.multiply(tf.tanh(cell_state),o,name='hidden_state')
                return [hidden_state,cell_state]

    def n_steps(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        with tf.name_scope(self.name_scope):
            input_shape = input_tensor.shape.as_list()
            if previous_cell_state is None:
                #previous_cell_state = tf.zeros_like(tf.matmul(input_tensor[:,0,:],self.Wx_i),dtype=tf.float32,name='init_cell_state')
                previous_cell_state = tf.zeros(input_tensor.shape[:1].concatenate(self.output_size),dtype=tf.float32,name='init_cell_state')
            if previous_hidden_state is None:
                previous_hidden_state = tf.zeros_like(previous_cell_state,dtype=tf.float32,name='init_hidden_state')

            hidden_state_list = []
            for i in range(input_shape[-2]):
                previous_hidden_state,previous_cell_state = self.one_step(input_tensor[:,i,:],previous_hidden_state,previous_cell_state)
                hidden_state_list.append(previous_hidden_state)
            hidden_state = tf.stack(hidden_state_list,axis=1,name='multiple_hidden_states')
            return hidden_state

    def one_step_new(self,input_tensor,hidden_states,previous_cell_state):
        with tf.name_scope(self.name_scope):
            with tf.name_scope('one_step'):
                previous_hidden_state = hidden_states[:,-1]
                li = tf.add(self.b_i,tf.add(tf.matmul(input_tensor,self.Wx_i),tf.matmul(previous_hidden_state,self.Wh_i)))
                lo = tf.add(self.b_o,tf.add(tf.matmul(input_tensor,self.Wx_o),tf.matmul(previous_hidden_state,self.Wh_o)))
                lf = tf.add(self.b_f,tf.add(tf.matmul(input_tensor,self.Wx_f),tf.matmul(previous_hidden_state,self.Wh_f)))
                lg = tf.add(self.b_g,tf.add(tf.matmul(input_tensor,self.Wx_g),tf.matmul(previous_hidden_state,self.Wh_g)))
                if self.peephole_flag:
                    li = tf.add(li,tf.matmul(previous_cell_state,self.Wc_i))
                    lo = tf.add(lo,tf.matmul(previous_cell_state,self.Wc_o))
                    lf = tf.add(lf,tf.matmul(previous_cell_state,self.Wc_f))
                    lg = tf.add(lg,tf.matmul(previous_cell_state,self.Wc_g))
                i = tf.sigmoid(li,name='i')
                o = tf.sigmoid(lo,name='o')
                f = tf.sigmoid(lf,name='f')
                g = tf.tanh(lg,name='g')
                cell_state = tf.add(tf.multiply(previous_cell_state,f),tf.multiply(g,i),name='cell_state')
                hidden_state = tf.multiply(tf.tanh(cell_state),o,name='hidden_state')
                hidden_states = tf.concat([hidden_states,tf.stack([hidden_state],axis=1)],axis=1)
                return [cell_state,hidden_states]

    def n_steps_new(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        with tf.name_scope(self.name_scope):
            if previous_cell_state is None:
                previous_cell_state = tf.zeros(tf.concat([tf.shape(input_tensor)[:1],[self.output_size,]],axis=0),dtype=tf.float32,name='init_cell_state')
            if previous_hidden_state is None:
                previous_hidden_state = tf.zeros_like(previous_cell_state,dtype=tf.float32,name='init_hidden_state')
            hidden_states = tf.stack([previous_hidden_state],axis=1)
            _,_,_,hidden_states = tf.while_loop(lambda cnt,input_tensor,cell_state,hidden_states: cnt<tf.shape(input_tensor)[-2],
                          lambda cnt,input_tensor,cell_state,hidden_states: [cnt+1,input_tensor]+self.one_step_new(tf.slice(input_tensor,tf.concat([tf.zeros([1],tf.int32),tf.reshape(cnt,[1]),tf.zeros([1],tf.int32)],axis=0),[-1,1,-1])[:,0],hidden_states,cell_state),
                          loop_vars = [tf.zeros([],tf.int32),input_tensor,previous_cell_state,hidden_states],
                          shape_invariants = [tf.TensorShape([]),input_tensor.get_shape(),previous_cell_state.get_shape(),input_tensor.shape[:1].concatenate(tf.TensorShape([None,self.output_size]))]
                          )
            hidden_states = tf.slice(hidden_states,[0,1,0],[-1]*3,name='multiple_hidden_states')
            return hidden_states

    def get_l2_loss(self):
        loss = tf.reduce_mean(tf.square(self.Wx_i))
        loss = loss + tf.reduce_mean(tf.square(self.Wx_o))
        loss = loss + tf.reduce_mean(tf.square(self.Wx_f))
        loss = loss + tf.reduce_mean(tf.square(self.Wx_g))
        loss = loss + tf.reduce_mean(tf.square(self.Wh_i))
        loss = loss + tf.reduce_mean(tf.square(self.Wh_o))
        loss = loss + tf.reduce_mean(tf.square(self.Wh_f))
        loss = loss + tf.reduce_mean(tf.square(self.Wh_g))
        if self.peephole_flag:
            loss = loss + tf.reduce_mean(tf.square(self.Wh_i))
            loss = loss + tf.reduce_mean(tf.square(self.Wh_o))
            loss = loss + tf.reduce_mean(tf.square(self.Wh_f))
            loss = loss + tf.reduce_mean(tf.square(self.Wh_g))
        return loss

class LSTMLNCell:
    '''
    The class for the LSTMLN layer, which contains the weight matrices and biases and one function to generate LSTMLN output
    The input neuron number is input_size, the output neuron number is output size. (the time step length is not needed)
    The input_tensor should be tensor of shape [N,T,input_size], where N is the batch size, T is the time step length.
    Wx_* are the weight matrices of shape [input_size,output_size], Wh_* are the weight matrices of shape [output_size,output_size] and b_* are the bias vector of shape [output_size].
    '''
    def __init__(self,input_size,output_size,name_scope='LSTMCell',summ_flag=True,
                 Wx_i=None,Wx_i_init=None,Wx_o=None,Wx_o_init=None,Wx_f=None,Wx_f_init=None,Wx_g=None,Wx_g_init=None,
                 Wh_i=None,Wh_i_init=None,Wh_o=None,Wh_o_init=None,Wh_f=None,Wh_f_init=None,Wh_g=None,Wh_g_init=None,
                 b_i=None,b_i_init=None,b_o=None,b_o_init=None,b_f=None,b_f_init=None,b_g=None,b_g_init=None,
                 gain_i=None,gain_i_init=None,gain_o=None,gain_o_init=None,gain_f=None,gain_f_init=None,gain_g=None,gain_g_init=None,
                 bias_i=None,bias_i_init=None,bias_o=None,bias_o_init=None,bias_f=None,bias_f_init=None,bias_g=None,bias_g_init=None,
                 gain_cell=None,gain_cell_init=None,bias_cell=None,bias_cell_init=None):
        self.input_size = input_size
        self.output_size = output_size
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            self.layernorm_i = LayerNorm(output_size,gain=gain_i,gain_init=gain_i_init,name_scope=self.name_scope+'LayerNorm_input',summ_flag=summ_flag)
            self.layernorm_o = LayerNorm(output_size,gain=gain_o,gain_init=gain_o_init,name_scope=self.name_scope+'LayerNorm_output',summ_flag=summ_flag)
            self.layernorm_f = LayerNorm(output_size,gain=gain_f,gain_init=gain_f_init,name_scope=self.name_scope+'LayerNorm_forget',summ_flag=summ_flag)
            self.layernorm_g = LayerNorm(output_size,gain=gain_g,gain_init=gain_g_init,name_scope=self.name_scope+'LayerNorm_g',summ_flag=summ_flag)
            self.layernorm_cell = LayerNorm(output_size,gain=gain_cell,gain_init=gain_cell_init,name_scope=self.name_scope+'LayerNorm_cell',summ_flag=summ_flag)
            with tf.name_scope('initialize_variables'):
                self.Wx_i = init_random_variable(Wx_i,Wx_i_init,[input_size,output_size],1./(input_size+output_size),'Wx_i')
                self.Wx_o = init_random_variable(Wx_o,Wx_o_init,[input_size,output_size],1./(input_size+output_size),'Wx_o')
                self.Wx_f = init_random_variable(Wx_f,Wx_f_init,[input_size,output_size],1./(input_size+output_size),'Wx_f')
                self.Wx_g = init_random_variable(Wx_g,Wx_g_init,[input_size,output_size],1./(input_size+output_size),'Wx_g')
                self.Wh_i = init_random_variable(Wh_i,Wh_i_init,[output_size,output_size],1./(input_size+output_size),'Wh_i')
                self.Wh_o = init_random_variable(Wh_o,Wh_o_init,[output_size,output_size],1./(input_size+output_size),'Wh_o')
                self.Wh_f = init_random_variable(Wh_f,Wh_f_init,[output_size,output_size],1./(input_size+output_size),'Wh_f')
                self.Wh_g = init_random_variable(Wh_g,Wh_g_init,[output_size,output_size],1./(input_size+output_size),'Wh_g')
                self.b_i = init_zero_variable(b_i,b_i_init,[output_size],'b_i')
                self.b_o = init_zero_variable(b_o,b_o_init,[output_size],'b_o')
                self.b_f = init_one_variable(b_f,b_f_init,[output_size],'b_f')
                self.b_g = init_zero_variable(b_g,b_g_init,[output_size],'b_g')
                if summ_flag:
                    self.Wx_i_summ = tf.summary.histogram('Wx_i',self.Wx_i)
                    self.Wx_o_summ = tf.summary.histogram('Wx_o',self.Wx_o)
                    self.Wx_f_summ = tf.summary.histogram('Wx_f',self.Wx_f)
                    self.Wx_g_summ = tf.summary.histogram('Wx_g',self.Wx_g)
                    self.Wh_i_summ = tf.summary.histogram('Wh_i',self.Wh_i)
                    self.Wh_o_summ = tf.summary.histogram('Wh_o',self.Wh_o)
                    self.Wh_f_summ = tf.summary.histogram('Wh_f',self.Wh_f)
                    self.Wh_g_summ = tf.summary.histogram('Wh_g',self.Wh_g)
                    self.b_i_summ = tf.summary.histogram('b_i',self.b_i)
                    self.b_o_summ = tf.summary.histogram('b_o',self.b_o)
                    self.b_f_summ = tf.summary.histogram('b_f',self.b_f)
                    self.b_g_summ = tf.summary.histogram('b_g',self.b_g)

    def __del__(self):
        del self.input_size
        del self.output_size
        del self.name_scope
        del self.Wx_i
        del self.Wx_o
        del self.Wx_f
        del self.Wx_g
        del self.Wh_i
        del self.Wh_o
        del self.Wh_f
        del self.Wh_g
        del self.b_i
        del self.b_o
        del self.b_f
        del self.b_g
        del self.layernorm_i
        del self.layernorm_o
        del self.layernorm_f
        del self.layernorm_g
        del self.layernorm_cell
        if self.summ_flag:
            del self.Wx_i_summ
            del self.Wx_o_summ
            del self.Wx_f_summ
            del self.Wx_g_summ
            del self.Wh_i_summ
            del self.Wh_o_summ
            del self.Wh_f_summ
            del self.Wh_g_summ
            del self.b_i_summ
            del self.b_o_summ
            del self.b_f_summ
            del self.b_g_summ
        del self.summ_flag

    def __call__(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        return self.n_steps(input_tensor,previous_hidden_state,previous_cell_state)

    def one_step(self,input_tensor,previous_hidden_state,previous_cell_state):
        with tf.name_scope(self.name_scope):
            with tf.name_scope('one_step'):
                i = tf.sigmoid(self.layernorm_i(tf.add(self.b_i,tf.add(tf.matmul(input_tensor,self.Wx_i),tf.matmul(previous_hidden_state,self.Wh_i)))),name='i')
                o = tf.sigmoid(self.layernorm_o(tf.add(self.b_o,tf.add(tf.matmul(input_tensor,self.Wx_o),tf.matmul(previous_hidden_state,self.Wh_o)))),name='o')
                f = tf.sigmoid(self.layernorm_f(tf.add(self.b_f,tf.add(tf.matmul(input_tensor,self.Wx_f),tf.matmul(previous_hidden_state,self.Wh_f)))),name='f')
                g = tf.tanh(self.layernorm_g(tf.add(self.b_g,tf.add(tf.matmul(input_tensor,self.Wx_g),tf.matmul(previous_hidden_state,self.Wh_g)))),name='g')
                cell_state = tf.add(tf.multiply(previous_cell_state,f),tf.multiply(g,i),name='cell_state')
                hidden_state = tf.multiply(tf.tanh(self.layernorm_cell(cell_state)),o,name='hidden_state')
                return (hidden_state,cell_state)

    def n_steps(self,input_tensor,previous_hidden_state=None,previous_cell_state=None):
        with tf.name_scope(self.name_scope):
            input_shape = input_tensor.shape.as_list()
            if previous_cell_state is None:
                previous_cell_state = tf.zeros_like(tf.matmul(input_tensor[:,0,:],self.Wx_i),dtype=tf.float32,name='init_cell_state')
            if previous_hidden_state is None:
                previous_hidden_state = tf.zeros_like(previous_cell_state,dtype=tf.float32,name='init_hidden_state')

            hidden_state_list = []
            for i in range(input_shape[-2]):
                previous_hidden_state,previous_cell_state = self.one_step(input_tensor[:,i,:],previous_hidden_state,previous_cell_state)
                hidden_state_list.append(previous_hidden_state)
            hidden_state = tf.stack(hidden_state_list,axis=1,name='multiple_hidden_states')
            return hidden_state

    def get_l2_loss(self):
        loss = tf.reduce_mean(tf.square(self.Wx_i))
        loss = loss + tf.reduce_mean(tf.square(self.Wx_o))
        loss = loss + tf.reduce_mean(tf.square(self.Wx_f))
        loss = loss + tf.reduce_mean(tf.square(self.Wx_g))
        loss = loss + tf.reduce_mean(tf.square(self.Wh_i))
        loss = loss + tf.reduce_mean(tf.square(self.Wh_o))
        loss = loss + tf.reduce_mean(tf.square(self.Wh_f))
        loss = loss + tf.reduce_mean(tf.square(self.Wh_g))
        return loss

def MLSTMlayer(input_dim1,input_dim2,output_dim,input_data1,input_data2,name1='MLSTM1',name2='MLSTM2'):
    lstm1 = LSTMCell(input_dim1,output_dim,name1)
    lstm2 = LSTMCell(input_dim2,output_dim,name2,Wh_i=lstm1.Wh_i,Wh_o=lstm1.Wh_o,Wh_g=lstm1.Wh_g,Wh_f=lstm1.Wh_f)
    return lstm1(input_data1),lstm2(input_data2),lstm1,lstm2

def LayerNormDropLayer(input_tensor,norm_shape,norm_axis,drop_rate,train_flag,norm_indic=1,drop_indic=1,activation_func=tf.nn.relu,name_scope='NormDropLayer',summ_flag=False):
    with tf.name_scope(name_scope):
        if norm_indic == 1:
            norm = LayerNorm(norm_shape,name_scope='LayerNorm',summ_flag=summ_flag)
            norm_out = activation_func(norm(input_tensor,norm_axis),name='norm_out')
        else:
            norm_out = activation_func(input_tensor,name='norm_out')
        if drop_indic == 1:
            return tf.layers.dropout(norm_out,rate=drop_rate,training=train_flag,name='drop_out')
        else:
            return norm_out

def numpyLayerNorm(input_tensor,gain=1,bias=0):
    return np.multiply(np.divide(np.subtract(input_tensor,np.mean(input_tensor,axis=-1,keepdims=True)),np.std(input_tensor,axis=-1,keepdims=True)),gain)+bias

def numpyLSTMLN(Wx_i,Wx_o,Wx_f,Wx_g,Wh_i,Wh_o,Wh_f,Wh_g,b_i,b_o,b_f,b_g,X):
    def sigmoid(x):
        return 1./(1.+np.exp(-x))
    N,T,D = X.shape
    hidden_state = np.zeros([N,3])
    cell_state = np.zeros([N,3])
    hidden_state_list = []
    for i in range(T):
        lstm_i = sigmoid(numpyLayerNorm(np.matmul(X[:,i,:],Wx_i)+np.matmul(hidden_state,Wh_i)+b_i))
        lstm_o = sigmoid(numpyLayerNorm(np.matmul(X[:,i,:],Wx_o)+np.matmul(hidden_state,Wh_o)+b_o))
        lstm_f = sigmoid(numpyLayerNorm(np.matmul(X[:,i,:],Wx_f)+np.matmul(hidden_state,Wh_f)+b_f))
        lstm_g = np.tanh(numpyLayerNorm(np.matmul(X[:,i,:],Wx_g)+np.matmul(hidden_state,Wh_g)+b_g))
        cell_state = lstm_f*cell_state+lstm_i*lstm_g
        hidden_state = np.tanh(numpyLayerNorm(cell_state))*lstm_o
        hidden_state_list.append(hidden_state)
    return np.stack(hidden_state_list,axis=1)

def numpyLSTM(Wx_i,Wx_o,Wx_f,Wx_g,Wh_i,Wh_o,Wh_f,Wh_g,b_i,b_o,b_f,b_g,X):
    def sigmoid(x):
        return 1./(1.+np.exp(-x))
    N,T,D = X.shape
    hidden_state = np.zeros([N,3])
    cell_state = np.zeros([N,3])
    hidden_state_list = []
    for i in range(T):
        lstm_i = sigmoid(np.matmul(X[:,i,:],Wx_i)+np.matmul(hidden_state,Wh_i)+b_i)
        lstm_o = sigmoid(np.matmul(X[:,i,:],Wx_o)+np.matmul(hidden_state,Wh_o)+b_o)
        lstm_f = sigmoid(np.matmul(X[:,i,:],Wx_f)+np.matmul(hidden_state,Wh_f)+b_f)
        lstm_g = np.tanh(np.matmul(X[:,i,:],Wx_g)+np.matmul(hidden_state,Wh_g)+b_g)
        cell_state = lstm_f*cell_state+lstm_i*lstm_g
        hidden_state = np.tanh(cell_state)*lstm_o
        hidden_state_list.append(hidden_state)
    return np.stack(hidden_state_list,axis=1)

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

def LSTMBlock(input_tensor,hidden_size,train_flag=None,gated_flag=False,ln_flag=True,dp_flag=False,dp=0.5,peephole_flag=True,activation_func=tf.nn.relu,name_scope='LSTMBlock'):
    with tf.name_scope(name_scope):
        lstm_layer = LSTMCell(input_tensor.shape.as_list()[-1],hidden_size*(2 if gated_flag else 1),name_scope='LSTMLayer')
        lstm_out = lstm_layer(input_tensor)
        if gated_flag:
            lstm_out_candidate = _get_first_half_channel(lstm_out)
            if ln_flag:
                lstm_candidate_ln_layer = LayerNorm([hidden_size],name_scope='lstm_candidate_layer_norm_layer')
                lstm_out_candidate = lstm_candidate_ln_layer(lstm_out_candidate)
            lstm_out_candidate = tf.tanh(lstm_out_candidate)
            lstm_out_gate = _get_last_half_channel(lstm_out)
            if ln_flag:
                lstm_gate_ln_layer = LayerNorm([hidden_size],name_scope='lstm_gate_layer_norm_layer')
                lstm_out_gate = lstm_gate_ln_layer(lstm_out_gate)
            lstm_out_gate = tf.sigmoid(lstm_out_gate)
            lstm_out = tf.multiply(lstm_out_candidate,lstm_out_gate,name='lstm_gated_out')
        else:
            if ln_flag:
                lstm_ln_layer = LayerNorm([hidden_size],name_scope='lstm_layer_norm_layer')
                lstm_out = lstm_ln_layer(lstm_out)
            lstm_out = activation_func(lstm_out,name='lstm_activated_out')
        if dp_flag:
            lstm_out = tf.layers.dropout(lstm_out,dp,training=train_flag,name='lstm_out_drop')
    return lstm_layer,lstm_out

def stackedLSTMBlock(input_tensor,hidden_size,layer_number,train_flag=None,gated_flag=False,ln_flag=True,dp_flag=False,dp=0.5,res_flag=True,skip_flag=True,peephole_flag=True,activation_func=tf.nn.relu,name_scope='stackedLSTMBlock'):
    with tf.name_scope(name_scope):
        stacked_lstm_layers = []
        stacked_lstm_input = [input_tensor]
        stacked_lstm_output = []
        for layer_num in range(1,layer_number+1):
            lstm_layer,lstm_out = LSTMBlock(stacked_lstm_input[-1],hidden_size,train_flag=train_flag,gated_flag=gated_flag,ln_flag=ln_flag,dp_flag=dp_flag,dp=dp,peephole_flag=peephole_flag,activation_func=activation_func,name_scope='LSTMBlock%d'%layer_num)
            stacked_lstm_layers.append(lstm_layer)
            stacked_lstm_output.append(lstm_out)
            if res_flag and layer_num > 1:
                stacked_lstm_input.append(tf.add(stacked_lstm_input[-1],lstm_out,name='lstm_residual_out%d'%layer_num))
            else:
                stacked_lstm_input.append(lstm_out)
        if skip_flag:
            stacked_lstm_out = tf.add_n(stacked_lstm_output,name='stacked_lstm_out')
        else:
            stacked_lstm_out = stacked_lstm_input[-1]
    return stacked_lstm_out, stacked_lstm_layers,stacked_lstm_output
