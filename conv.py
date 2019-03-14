#!/usr/bin/env python
# coding=utf-8

# 20180805 by tanghao
# reorganized from nn_parts.py
# This file contains Convolution-related layers

import tensorflow as tf
import numpy as np
from .init_var import *

class ConvCell:
    '''
    The class for the formal Convolution2D layer without bias, which contains the filters and parameters
    '''
    def __init__(self,filter_size,strides=[1,1,1,1],padding='SAME',data_format='NCHW',name_scope='ConvCell'\
                 ,F=None,F_init=None,summ_flag=True):
        '''
        filter_size is of type list and length 4, which is [filter_height, filter_width, in_channels, out_channels]
        strides is a list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. The dimension order is determined by the value of data_format, see below for details.
        padding is a string from: "SAME", "VALID". The type of padding algorithm to use. "SAME" will pad the same value, and "VALID" will not pad
        data_format is an optional string from: "NHWC", "NCHW". Defaults to "NHWC". Specify the data format of the input and output data.
            With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels].
            Alternatively, the format could be "NCHW", the data storage order of: [batch, channels, height, width].
        name_scope should be of type string
        F is tf.Variable with shape equal to self.filter_size
        init_F can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape self.filter_size
        summ_flag is boolean, indicating whether tensors are summarized
        '''
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            self.F = init_random_variable(F,F_init,self.filter_size,np.sqrt(2./(self.filter_size[0]*self.filter_size[1]*self.filter_size[2])),'filter')
            if self.summ_flag:
                self.F_summ = tf.summary.histogram('F_summ',self.F)
    def __del__(self,):
        if self.summ_flag:
            del self.F_summ
        del self.filter_size
        del self.strides
        del self.padding
        del self.data_format
        del self.name_scope
        del self.F
        del self.summ_flag

    def get_l2_loss(self,):
        return tf.reduce_mean(tf.square(self.F))
    def __call__(self,input_tensor):
        '''
        If self.default format is "NHWC",
            the input_tensor is stored in the order of: [batch, height, width, input_channels].
            return output_tensor of shape [batch, height, width, output_channels].
        Alternatively, if the format is "NCHW",
            the input_tensor storage order of: [batch, input_channels, height, width].
            return output_tensor of shape [batch, output_channels, height, width].
        '''
        return self.get_output(input_tensor)
    def get_output(self,input_tensor):
        '''
        If self.default format is "NHWC",
            the input_tensor is stored in the order of: [batch, height, width, input_channels].
            return output_tensor of shape [batch, height, width, output_channels].
        Alternatively, if the format is "NCHW",
            the input_tensor storage order of: [batch, input_channels, height, width].
            return output_tensor of shape [batch, output_channels, height, width].
        '''
        with tf.name_scope(self.name_scope):
            return tf.nn.conv2d(input_tensor,self.F,self.strides,self.padding,data_format=self.data_format,name='conv2d_out')

class CausalDilatedConvCell:
    '''
    The class for the causal dilated convolution layer (firstly appear in wavenet), which contains the filters and parameters
    '''
    def __init__(self,filter_size,dilation_rate,bias_flag=True,name_scope='ConvCell',F=None,F_init=None,b=None,b_init=None,summ_flag=True):
        '''
        Wrapper of tf.convolution for causal dilated convolution implementation,
        Assume the data_format is 'NWC', and with shape [num_batches, input_spatial_shape[0], ..., input_spatial_shape[N-1], num_input_channels], and the input_spatial_shape[0] is the temporal dimensionality
            filter_size is of type list and is [spatial_filter_shape[0], ..., spatial_filter_shape[N-1], num_input_channels, num_output_channels]
            dilation_rate is sequences of N-2 ints>=1 Specifies the filter upsampling/input downsampling rate.
                In the literature, the same parameter is sometimes called input stride or dilation. The effective filter size used for the convolution will be spatial_filter_shape + (spatial_filter_shape - 1) * (rate - 1), obtained by inserting (dilation_rate[i]-1) zeros between consecutive elements of the original filter in each spatial dimension i.
                dilation_rate[0] and dilation_rate[-1] must be 1
            bias_flag indicates the existence of bias, and is of type boolean
            name_scope should be of type string
            F specifies the filter variable, and should be of type tf.Variable with shape equal to self.filter_size, defaults is None
            F_init Specifies the initial value of the filter variable, and can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape self.filter_size
            b specifies the bias variable, and should be of type tf.Variable with shape equal to self.filter_size[-1], defaults is None
            b_init Specifies the initial value of the bias variable, and can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape self.filter_size[-1]
            summ_flag is boolean, indicating whether tensors are summarized
        '''
        self.filter_size = [int(f) for f in filter_size]
        self.dilation_rate = [int(d) for d in dilation_rate]
        self.summ_flag = summ_flag
        self.bias_flag = bias_flag
        with tf.name_scope(name_scope) as self.name_scope:
            self.F = init_random_variable(F,F_init,self.filter_size,1./np.prod(self.filter_size[:-1]),'filter')
            self.b = init_zero_variable(b,b_init,self.filter_size[-1],'bias')
            if self.summ_flag:
                self.F_summ = tf.summary.histogram('F_summ',self.F)
                self.b_summ = tf.summary.histogram('b_summ',self.b)
    def get_l2_loss(self,):
        return tf.reduce_sum(tf.square(self.F))
    def __call__(self,input_tensor):
        '''
        Assume the data_format is 'NWC', and with shape [num_batches, input_spatial_shape[0], ..., input_spatial_shape[N-1], num_input_channels], and the input_spatial_shape[0] is the temporal dimensionality
        '''
        return self.get_output(input_tensor)
    def get_output(self,input_tensor):
        '''
        Assume the data_format is 'NWC', and with shape [num_batches, input_spatial_shape[0], ..., input_spatial_shape[N-1], num_input_channels], and the input_spatial_shape[0] is the temporal dimensionality
        '''
        with tf.name_scope(self.name_scope):
            padded_input_tensor = tf.pad(input_tensor,[(0,0)]+[(d*(self.filter_size[i]-1),0) for i,d in enumerate(self.dilation_rate)]+[(0,0)],mode='CONSTANT',name='padded_input_tensor',constant_values=0)
            conv_out = tf.nn.convolution(padded_input_tensor,self.F,'VALID',dilation_rate=self.dilation_rate,name='conv_out')
            cell_out = tf.add(conv_out,self.b,name='cell_out')
            return cell_out


if __name__ == '__main__':
    a = tf.zeros([10,8064,40])
    cdconv_layer1 = CausalDilatedConvCell([2,40,7],[1,1,1])
    cdconv_out1 = cdconv_layer1(a)
    cdconv_layer2 = CausalDilatedConvCell([2,7,3],[1,2,1])
    cdconv_out2 = cdconv_layer2(cdconv_out1)
    print(a.shape,cdconv_out1.shape,cdconv_out2.shape)

