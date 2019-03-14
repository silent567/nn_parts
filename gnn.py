#!/usr/bin/env python
# coding=utf-8

# 20180825 by tanghao
# This file contains graph-network-related layers

import tensorflow as tf
from .init_var import *
from .fc import *
from .norm import LayerNorm

class FastGCNN:
    '''
    The class for the FastGCNN layer, which is based on "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering"
    '''
    def __init__(self,kernel_size,input_filter_size,output_filter_size,node_num=None,dynamical_graph_flag=True,name_scope='FastGCNNCell',F=None,F_init=None,L=None,L_init=None,summ_flag=True):
        '''
        kernel_size is positive int, which is K in the paper, the polynomial degree of filters
        input_filter_size is int, which is input channel number
        output_filter_size is int, which is output channel number
        node_num is int, which is the vertex number in the graph.
        dynamical_graph_flag is boolean, which denotes whether the Laplacian Matrix is updated by the optimizer
        name_scope should be of type string
        F is tf.Variable with shape equal to [self.input_filter_size,self.output_filter_size,self.kernel_size]
        init_F can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape [self.input_filter_size,self.output_filter_size,self.kernel_size]
        L is tf.Variable with shape equal to [self.node_num,self.node_num]
        init_L can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape [self.node_num,self.node_num]
        summ_flag is boolean, indicating whether tensors are summarized

        One of node_num, L, L_init should not be None for Laplacian Matrix initialization

        Sample use:
            gcnn_layer = FastGCNN(kernel_size,input_filter_size,output_filter_size,node_num)
            gcnn_layer = FastGCNN(kernel_size,input_filter_size,output_filter_size,dynamical_graph_flag=False,L=LaplacianMatrix)
        '''
        self.input_filter_size = input_filter_size
        self.output_filter_size = output_filter_size
        self.kernel_size = kernel_size
        self.dynamical_graph_flag = dynamical_graph_flag
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            L = init_identity_matrix_variable(L,L_init,node_num,'UnsymmetrixLaplacianMatrix')
            self.L = tf.divide(L+tf.transpose(L),2.,name='LaplacianMatrix')
            self.node_num = self.L.shape.as_list()[-1]
            self.L_maxeigenvalue = tf.self_adjoint_eig(self.L)[0][-1]
            self.normL = tf.subtract(2*self.L/self.L_maxeigenvalue,tf.eye(self.node_num),name='NormedLaplacianMatrix')
            if self.kernel_size == 1:
                TnormL_list = [tf.eye(self.node_num)]
            else:
                TnormL_list = [tf.eye(self.node_num),self.normL]
                for tindex in range(2,self.kernel_size):
                    TnormL_list.append(2*tf.matmul(self.normL,TnormL_list[-1])-TnormL_list[-2])
            self.TnormL = tf.stack(TnormL_list,axis=0)

            self.F = init_random_variable(F,F_init,[self.input_filter_size,self.output_filter_size,self.kernel_size],2./(self.input_filter_size*self.node_num),'filter')
            self.coefficents = tf.einsum('aim,mjk->aijk',self.F,self.TnormL,name='coefficents')

            if self.summ_flag:
                self.F_summ = tf.summary.histogram('F_summ',self.F)
                self.L_summ = tf.summary.histogram('L_summ',self.L)
                self.normL_summ = tf.summary.histogram('normL_summ',self.normL)
    def get_l2_loss(self,):
        return tf.reduce_mean(tf.square(self.F))
    def __call__(self,input_tensor):
        return self.get_output(input_tensor)
    def get_output(self,input_tensor):
        '''
        input_tensor should be of shape [N,node_num,input_filter_size]
        output_tensor should be of the same type as input_tensor
            and of shape [N,node_num,output_filter_size]
        '''
        with tf.name_scope(self.name_scope):
            return tf.einsum('nai,ijab->nbj',input_tensor,self.coefficents)

class DenseUpdateLayer(object):
    def __init__(self,input_size,output_size,layer_num,norm_flag=True,dropout_flag=False,res_flag=True,activation_func=tf.nn.leaky_relu,summ_flag=False,name_scope='DenseUpdateLayer'):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_num = layer_num
        self.norm_flag = norm_flag
        self.dropout_flag = dropout_flag
        self.res_flag = res_flag
        self.activation_func = activation_func
        self.summ_flag = summ_flag
        with tf.name_scope(name_scope) as self.name_scope:
            pass
        self.build_model()

    def build_model(self):
        input_size = self.input_size
        output_size = self.output_size
        layer_num = self.layer_num
        summ_flag = self.summ_flag

        self.name_scope_layers = []
        self.dense_layers = []
        self.norm_layers = []
        with tf.name_scope(self.name_scope):
            for ln in range(layer_num):
                with tf.name_scope('Layer%d'%ln) as tmp_name_scope:
                    self.name_scope_layers.append(tmp_name_scope)
                    self.dense_layers.append(Dense(input_size,output_size,activation_func=linear_activation,summ_flag=summ_flag))
                    self.norm_layers.append(LayerNorm([output_size],summ_flag=summ_flag))
                    input_size = output_size

    def __call__(self,X,train_flag):
        '''
        input arguments:
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
        output: updated node attributes X' of type tf.Tensor annd of shape [N,C']
        '''
        norm_flag = self.norm_flag
        dropout_flag = self.dropout_flag
        res_flag = self.res_flag
        activation_func = self.activation_func

        with tf.name_scope(self.name_scope):
            input_X = X
            for ns,dense,norm in zip(self.name_scope_layers,self.dense_layers,self.norm_layers):
                with tf.name_scope(ns):
                    output_X = dense(input_X)
                    if norm_flag:
                        output_X = norm(output_X)
                    output_X = activation_func(output_X)
                    if dropout_flag:
                        output_X = tf.layers.dropout(output_X,0.5,training=train_flag)
                    if res_flag and dense.input_size == dense.output_size:
                        output_X = tf.add(output_X,input_X)
                    input_X = output_X
        return output_X

    def get_l2_loss(self):
        with tf.name_scope(self.name_scope):
            l2_loss = tf.add_n([dense.get_l2_loss() for dense in self.dense_layers])
        return l2_loss

class MPNNLayer:
    def __init__(self,update_func,aggregate_func,edge_label_num,name_scope='MPNNLayer'):
        with tf.name_scope(name_scope) as self.name_scope:
            '''
            update_func is applied to X to update node attributes individually (similar to conv when kernel size=1)
            aggregate_func receives A and X and output aggregated node attributes X'
            '''
            self.update_func = update_func
            self.aggregate_func = aggregate_func
            self.edge_label_num = edge_label_num
    def __call__(self,A,X,train_flag):
        '''
        input arguments:
            A is the graph adjacency matrix of type tf.Tensor and of shape [N,N,M]
            X is the node attributes matrix of type tf.Tensor and of shape [N,N,C]
            , where N is the number of nodes, M is the number of edge classes, and C is the channel number of node attributes
            train_flag is the flag for dropout layer of type tf.Tensor, of shape [] and of type tf.Boolean
        output arguments:
            updated and aggregated new node attributes X' of type tf.Tensor and of shape [N,N,C']
        '''
        update_func = self.update_func
        aggregate_func = self.aggregate_func
        with tf.name_scope(self.name_scope):
            output_X_list = []
            for en in range(self.edge_label_num):
                updated_X = update_func(X,train_flag)
                aggregated_X = aggregate_func(A[:,:,en],updated_X)
                output_X_list.append(aggregated_X)
            output_X = tf.add_n(output_X_list,name='output_X')
        return output_X

class SumAggregator:
    def __init__(self,name_scope='SumAggregator'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def __call__(self,A,X):
        '''
        input arguments:
            A is the graph adjacency matrix of type tf.Tensor and of shape [N,N]
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
            , where N is the number of nodes and C is the channel number of node attributes
        output arguments:
            aggregated new node attributes X' of type tf.Tensor and of shape [N,C]
        '''
        with tf.name_scope(self.name_scope):
            self_loop_A = tf.add(A,tf.eye(tf.shape(A)[0]),name='self_loop_A')
            output_X = tf.matmul(self_loop_A,X,name='output_X')
            return output_X

class MeanAggregator:
    def __init__(self,name_scope='MeanAggregator'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def __call__(self,A,X):
        '''
        input arguments:
            A is the graph adjacency matrix of type tf.Tensor and of shape [N,N]
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
            , where N is the number of nodes and C is the channel number of node attributes
        output arguments:
            aggregated new node attributes X' of type tf.Tensor and of shape [N,C]
        '''
        with tf.name_scope(self.name_scope):
            self_loop_A = tf.add(A,tf.eye(tf.shape(A)[0]),name='self_loop_A')
            output_X = tf.divide(tf.matmul(self_loop_A,X),tf.reduce_sum(self_loop_A,axis=-1,keepdims=True),name='output_X')
            return output_X

class MaxAggregator_old:
    def __init__(self,name_scope='MaxAggregator'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def __call__(self,A,X):
        '''
        input arguments:
            A is the graph adjacency matrix of type tf.Tensor and of shape [N,N]
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
            , where N is the number of nodes and C is the channel number of node attributes
        output arguments:
            aggregated new node attributes X' of type tf.Tensor and of shape [N,C]
        '''
        with tf.name_scope(self.name_scope):
            output_shape = X.get_shape()
            node_num = tf.shape(X,name='output_shape')[0]
            self_loop_A = tf.add(A,tf.eye(node_num),name='self_loop_A')
            flat_self_loop_A = tf.reshape(self_loop_A,[-1,1],name='flat_self_loop_A')
            tiled_X = tf.tile(X,[node_num,1],name='tiled_flat_X')
            flat_X_dot_A = tf.reshape(tiled_X*flat_self_loop_A - 1e4*(1-flat_self_loop_A),[node_num,node_num,-1],name='flat_X_dot_A')
            output_X = tf.reduce_max(flat_X_dot_A,axis=1,keepdims=False,name='output_X')
            output_X.set_shape(output_shape)
            return output_X

class MaxAggregator:
    def __init__(self,name_scope='MaxAggregator'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def _maximum_neighborhood(self,index,A,X,out):
        with tf.name_scope(self.name_scope):
            neigh = tf.boolean_mask(X,A[index])
            max_neigh = tf.reduce_max(neigh,keepdims=True,axis=0)
            out = tf.concat([out,max_neigh],axis=0)
        return out
    def __call__(self,A,X):
        '''
        input arguments:
            A is the graph adjacency matrix of type tf.Tensor and of shape [N,N]
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
            , where N is the number of nodes and C is the channel number of node attributes
        output arguments:
            aggregated new node attributes X' of type tf.Tensor and of shape [N,C]
        '''
        with tf.name_scope(self.name_scope):
            output_shape = X.get_shape()
            node_num = tf.shape(X,name='output_shape')[0]
            output_dim = int(output_shape[-1])
            self_loop_A = tf.add(A,tf.eye(node_num),name='self_loop_A')

            output_X = tf.zeros([0,output_dim])
            _,_,_,output_X = tf.while_loop(lambda index,A,X,out: index<node_num,\
                          lambda index,A,X,out: [index+1,A,X,self._maximum_neighborhood(index,A,X,out)],\
                          loop_vars = [tf.zeros([],tf.int32),self_loop_A,X,output_X],\
                          shape_invariants = [tf.TensorShape([]),A.get_shape(),X.get_shape(),tf.TensorShape([None,output_dim])])
            output_X.set_shape(output_shape)
            return output_X

class GCNAggregator:
    def __init__(self,name_scope='GCNAggregator'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def __call__(self,A,X):
        '''
        input arguments:
            A is the graph adjacency matrix of type tf.Tensor and of shape [N,N]
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
            , where N is the number of nodes and C is the channel number of node attributes
        output arguments:
            aggregated new node attributes X' of type tf.Tensor and of shape [N,C]
        '''
        with tf.name_scope(self.name_scope):
            self_loop_A = tf.add(A,tf.eye(tf.shape(A)[0]),name='self_loop_A')
            self_loop_D_sqrt = tf.linalg.diag(1./tf.sqrt(tf.reduce_sum(self_loop_A,axis=1)),name='self_loop_D_sqrt')
            normalized_self_loop_A = tf.matmul(self_loop_D_sqrt,tf.matmul(self_loop_A,self_loop_D_sqrt),name='normalized_self_loop_A')
            output_X = tf.matmul(normalized_self_loop_A,X,name='output_X')
            return output_X

class SumGraphAggregator:
    def __init__(self,name_scope='SumGraphAggregator'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def __call__(self,X):
        '''
        input arguments:
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
            , where N is the number of nodes and C is the channel number of node attributes
        output arguments:
            aggregated new node attributes X' of type tf.Tensor and of shape [1,C]
        '''
        with tf.name_scope(self.name_scope):
            output_X = tf.reduce_sum(X,axis=0,keepdims=True,name='output_X')
            return output_X

class MeanGraphAggregator:
    def __init__(self,name_scope='MeanGraphAggregator'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def __call__(self,X):
        '''
        input arguments:
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
            , where N is the number of nodes and C is the channel number of node attributes
        output arguments:
            aggregated new node attributes X' of type tf.Tensor and of shape [1,C]
        '''
        with tf.name_scope(self.name_scope):
            output_X = tf.reduce_mean(X,axis=0,keepdims=True,name='output_X')
            return output_X

class MaxGraphAggregator:
    def __init__(self,name_scope='MaxGraphAggregator'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def __call__(self,X):
        '''
        input arguments:
            X is the node attributes matrix of type tf.Tensor and of shape [N,C]
            , where N is the number of nodes and C is the channel number of node attributes
        output arguments:
            aggregated new node attributes X' of type tf.Tensor and of shape [1,C]
        '''
        with tf.name_scope(self.name_scope):
            output_X = tf.reduce_max(X,axis=0,keepdims=True,name='output_X')
            return output_X

class CreateSubgraph:
    def __init__(self,name_scope='CreateSubgraph'):
        with tf.name_scope(name_scope) as self.name_scope:
            pass
    def _remove_one_node(self,X,A):
        with tf.name_scope(self.name_scope):
            indices = tf.range(tf.shape(A)[0])
            indices = tf.random_shuffle(indices)[:-1]
            X = tf.gather(X,indices)
            A = tf.gather(tf.gather(A,indices),indices,axis=1)
            return X,A
    def __call__(self,X,A):
        with tf.name_scope(self.name_scope):
            return self._remove_one_node(X,A)

