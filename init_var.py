#!/usr/bin/env python
# coding=utf-8

# 20180805 by tanghao
# reorganized from nn_parts.py
# This file contains functions related to initialization

import tensorflow as tf

def init_random_variable(value,init_value,shape,std,name,dtype=tf.float32):
    '''
    initialize the variables
    if value exists, return the exact value
    else if init_value exists, return the variable with init_value as the initialize value
    else return the variable initialized randomly with mean 0, stddev equal to std.
    '''
    if value is not None:
        return value
    elif init_value is not None:
        return tf.Variable(init_value,dtype=dtype,name=name)
    else:
        return tf.Variable(tf.truncated_normal(shape,0,std),dtype=dtype,name=name)

def init_identity_matrix_variable(value,init_value,size,name,dtype=tf.float32):
    '''
    initialize the variables
    if value exists, return the exact value
    else if init_value exists, return the variable with init_value as the initialize value
    else return the variable initialized with identity matrix of shape [size,size].
    '''
    if value is not None:
        return value
    elif init_value is not None:
        return tf.Variable(init_value,dtype=dtype,name=name)
    else:
        return tf.Variable(tf.eye(size),dtype=dtype,name=name)

def init_ortho_random_variable(value,init_value,shape,std,name,dtype=tf.float32):
    '''
    initialize the variables orthogonally
    if value exists, return the exact value
    else if init_value exists, return the variable with init_value as the initialize value
    else return the variable initialized randomly with mean 0, stddev equal to std.
    '''
    if value is not None:
        return value
    elif init_value is not None:
        return tf.Variable(init_value,dtype=dtype,name=name)
    else:
        return tf.get_variable(name,shape,dtype=tf.float32,initializer=tf.orthogonal_initializer())

def init_zero_variable(value,init_value,shape,name,dtype=tf.float32):
    if value is not None:
        return value
    elif init_value is not None:
        return tf.Variable(init_value,dtype=dtype,name=name)
    else:
        return tf.Variable(tf.zeros(shape),dtype=dtype,name=name)

def init_one_variable(value,init_value,shape,name,dtype=tf.float32):
    if value is not None:
        return value
    elif init_value is not None:
        return tf.Variable(init_value,dtype=dtype,name=name)
    else:
        return tf.Variable(tf.ones(shape),dtype=dtype,name=name)
