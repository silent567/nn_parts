#!/usr/bin/env python
# coding=utf-8

# 20180805 by tanghao
# reorganized from nn_parts.py
# This file contains test-related codes

from . import *
import numpy as np

#Wx_i = np.random.rand(4,3)
#Wx_o = np.random.rand(4,3)
#Wx_f = np.random.rand(4,3)
#Wx_g = np.random.rand(4,3)
#Wh_i = np.random.rand(3,3)
#Wh_o = np.random.rand(3,3)
#Wh_f = np.random.rand(3,3)
#Wh_g = np.random.rand(3,3)
#b_i = np.random.rand(3)
#b_o = np.random.rand(3)
#b_f = np.random.rand(3)
#b_g = np.random.rand(3)
#gain = np.random.rand(3)
#bias = np.random.rand(3)

#a = tf.placeholder(tf.float32,[None,3,4],name='input')
#disc_a = discrete_data(a,100)
#print(disc_a)
#lstm = LSTMLNCell(4,3,Wx_i_init=Wx_i,Wx_o_init=Wx_o,Wx_f_init=Wx_f,Wx_g_init=Wx_g,Wh_i_init=Wh_i,Wh_o_init=Wh_o,Wh_f_init=Wh_f,Wh_g_init=Wh_g,b_i_init=b_i,b_o_init=b_o,b_f_init=b_f,b_g_init=b_g)
#lstm2 = ConvLSTMCell(4,1,4,7)
#lstm_out = lstm(a)
#lstm_out2 = lstm2(a)
#ln = LayerNorm(3,gain_init=gain,bias_init=bias)
#lstm_out_norm = ln(lstm_out)
#b = Dense(3,1,W_init=np.ones([3,1]),b_init=np.zeros([1,]),name_scope='Dense')(tf.reshape(lstm_out_norm,[-1,3]))
#print(a,lstm_out,lstm_out_norm,b)
#summ = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #input_a = np.random.rand(2,3,4)-0.5
    #np_lstm_out = numpyLSTMLN(Wx_i,Wx_o,Wx_f,Wx_g,Wh_i,Wh_o,Wh_f,Wh_g,b_i,b_o,b_f,b_g,input_a)
    #np_lstm_out_norm =numpyLayerNorm(np_lstm_out,gain=gain,bias=bias)
    #print('numpy_LSTMLN_result:',np_lstm_out)
    #print('lstm_out:',sess.run(lstm_out,feed_dict={a:input_a}))
    #print('numpy_LSTM_out_norm:',np_lstm_out_norm)
    #print('lstm_out_norm:',sess.run(lstm_out_norm,feed_dict={a:input_a}))
    #print('lstm_out2: ',sess.run(lstm_out2,feed_dict={a:input_a}))
    #print('c: ',sess.run(c,feed_dict={a:input_a}))

    #writer = tf.summary.FileWriter('./log',graph=sess.graph)
    #writer.add_summary(sess.run(summ,feed_dict={a:input_a}))

    a = tf.placeholder(tf.float32,[None,10])
    b = expand_data(a,256,axis=2)
    c = discrete_data(a,256,axis=2)
    d = tf.reduce_min(tf.reduce_sum(b*c,axis=2))

    aa = np.random.rand(1,10)
    print('a:',(aa-np.min(aa))/(np.max(aa)-np.min(aa)+1e-9)*2)
    print('b:',sess.run(b,feed_dict={a:aa}))
    print('c:',sess.run(c,feed_dict={a:aa}))
    print('d:',sess.run(d,feed_dict={a:aa}))
