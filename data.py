#!/usr/bin/env python
# coding=utf-8

# 20180805 by tanghao
# reorganized from nn_parts.py
# This file contains Convolution-related layers

import tensorflow as tf

def discrete_data(data,depth,name_scope='Discrete_Data',axis=-1,minimum=None,maximum=None):
    with tf.name_scope(name_scope):
        if minimum is None:
            minimum = tf.reduce_min(data)
        if maximum is None:
            maximum = tf.reduce_max(data)

        scaled_int_data = tf.cast(tf.round((data-minimum)/(maximum-minimum+1e-22)*(depth-1)),dtype=tf.int32,name='scaled_int_data')
        return tf.one_hot(scaled_int_data,depth,on_value=1.,off_value=0.,axis=axis,dtype=tf.float32,name='discreted_data')

def expand_data(data,depth,name_scope='Expand_Data',axis=-1,minimum=None,maximum=None):
    with tf.name_scope(name_scope):
        if minimum is None:
            minimum = tf.reduce_min(data)
        if maximum is None:
            maximum = tf.reduce_max(data)
        cent_data = (data-minimum)/(maximum-minimum+1e-9)*(depth-1)
        scaled_floor_data = tf.cast(tf.floor(cent_data),dtype=tf.int32,name='scaled_floor_data')
        scaled_ceil_data = tf.cast(tf.ceil(cent_data),dtype=tf.int32,name='scaled_ceil_data')
        floor_one_hot = tf.one_hot(scaled_floor_data,depth,on_value=1.,off_value=0.,axis=axis,dtype=tf.float32,name='floor_one_hot')
        ceil_one_hot = tf.one_hot(scaled_ceil_data,depth,on_value=1.,off_value=0.,axis=axis,dtype=tf.float32,name='ceil_one_hot')
        data_exp = tf.expand_dims(cent_data,axis=axis,name='data_exp')
        data_floor = data_exp - tf.floor(data_exp,name='data_floor')
        print(scaled_floor_data.shape,scaled_ceil_data.shape,floor_one_hot.shape,ceil_one_hot.shape,data_exp.shape,data_floor.shape)
        return floor_one_hot*(1-data_floor)+ceil_one_hot*data_floor
