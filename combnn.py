#!/usr/bin/env python
# coding=utf-8

import tensorflow as _tf
from .init_var import *
from .conv import ConvCell
from .fc import Dense,linear_activation
from .norm import LayerNorm

class Node2Edge(object):
    '''
    Encode Node attributes to edge by binary multiplication.
    Assume X is node attributes of shape [N,C], A is adjacency matrix of shape [N,N], w is weight matrix of shape [C,C],
    the edge attribute e is calculated as e = (XwX^T+b) * A
    There are C' weight matrices corresponding to the output edge attributes E of shape [N,C']
    '''
    def __init__(self,input_size,output_size,name_scope='Node2Edge',mask_flag=True,summ_flag=True,bias_flag=True\
                 ,W=None,W_init=None,b=None,b_init=None):
        '''
        The input_size is the channel number of node attributes
        The output_size is the channel number of output edge attributes
        mask_flag is boolean, indicating whether edge attributes are preserved when no edge exists
        summ_flag is boolean, indicating whether tensors are summarized
        bias_flag is boolean, indicating whether biases are added
        W is tf.Variable of shape [output_size,input_size,input_size]
        b is tf.Variable with shape equal to [output_size,]
        init_W can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape [output_size,input_size,input_size]
        init_b can be tf.Variable, tf.Tensor, list, numpy.ndarray of shape [output_size,]
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.mask_flag = mask_flag
        self.summ_flag = summ_flag
        self.bias_flag = bias_flag
        with _tf.name_scope(name_scope) as self.name_scope:
            self.weights = init_random_variable(W,W_init,[output_size,input_size,input_size],1e-3,'random_weights')
            self.weights = _tf.divide((self.weights + _tf.transpose(self.weights,[0,2,1])),2.,name='weights')
            self.bias = init_zero_variable(b,b_init,[output_size,],name='bias')
    def __call__(self,X,A):
        '''
        input arguments;
            A is the graph adjacency matrix of type tf.Tensor and of shape [N,N]
            X is the node attributes matrix of type tf.Tensor and of shape [N,input_size]
            , where N is the number of nodes, and input_size is the channel number of node attributes
        output:
            E is the edge attributes matrix of type tf.Tensor and of shape [output_size,N,N]
            , where output_size is the channel number of edge attributes
        '''
        with _tf.name_scope(self.name_scope):
            out = _tf.stack([_tf.matmul(_tf.matmul(X,self.weights[i]),X,transpose_b=True) for i in range(self.output_size)],axis=0)
            if self.bias_flag:
                out = out + _tf.expand_dims(_tf.expand_dims(self.bias,axis=-1),axis=-1)
            if self.mask_flag:
                out = out * _tf.expand_dims(A,axis=0)
            return out
    def get_l2_loss(self,):
        with _tf.name_scope(self.name_scope):
            return _tf.reduce_sum(_tf.square(self.weights))

class CombineSubgraphs_old(object):
    def __init__(self,input_size,output_size,layer_num,norm_flag=True,dropout_flag=False,res_flag=True\
                 ,activation_func=_tf.nn.leaky_relu,summ_flag=False,name_scope='CombineSubgraphs'):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_num = layer_num
        self.norm_flag = norm_flag
        self.dropout_flag = dropout_flag
        self.res_flag = res_flag
        self.activation_func = activation_func
        self.summ_flag = summ_flag
        with _tf.name_scope(name_scope) as self.name_scope:
            pass
        self.build_model()
    def build_model(self):
        input_size = self.input_size
        output_size = self.output_size
        layer_num = self.layer_num
        summ_flag = self.summ_flag

        self.name_scope_layers = []
        self.conv_layers = []
        self.norm_layers = []
        with _tf.name_scope(self.name_scope):
            with _tf.name_scope('Transpose') as self.transpose_ns:
                pass
            with _tf.name_scope('Layer%d'%0) as tmp_name_scope:
                self.name_scope_layers.append(tmp_name_scope)
                self.conv_layers.append(ConvCell([2,1,input_size,output_size],padding='VALID',data_format='NHWC',summ_flag=self.summ_flag))
                self.norm_layers.append(LayerNorm([output_size,],summ_flag=summ_flag))
            for ln in range(1,layer_num):
                with _tf.name_scope('Layer%d'%ln) as tmp_name_scope:
                    self.name_scope_layers.append(tmp_name_scope)
                    self.conv_layers.append(ConvCell([1,1,output_size,output_size],padding='VALID',data_format='NHWC',summ_flag=self.summ_flag))
                    self.norm_layers.append(LayerNorm([output_size,],summ_flag=summ_flag))
            with _tf.name_scope('TransposeBack') as self.transpose_back_ns:
                pass
    def __call__(self,subgraph_repres,train_flag):
        '''
        input arguments:
            subgraph_repres is the subgraphs' representations of type tf.Tensor and of shape [1,C,N-index+1,1]
            closer subgraphs in subgraph_repres share more common nodes.
        output: combined subgraph_repres of type tf.Tensor annd of shape [1,C,N-index,1]
        '''
        norm_flag = self.norm_flag
        dropout_flag = self.dropout_flag
        res_flag = self.res_flag
        activation_func = self.activation_func

        with _tf.name_scope(self.name_scope):
            with _tf.name_scope(self.transpose_ns):
                input_repre = _tf.transpose(subgraph_repres,[0,2,3,1])
            for ns,conv,norm in zip(self.name_scope_layers,self.conv_layers,self.norm_layers):
                with _tf.name_scope(ns):
                    output_repre = conv(input_repre)
                    if norm_flag:
                        output_repre = norm(output_repre)
                    output_repre = activation_func(output_repre)
                    if dropout_flag:
                        output_repre = _tf.layers.dropout(output_repre,0.5,training=train_flag)
                    if res_flag and conv.filter_size[0] == 1:
                        output_repre = _tf.add(output_repre,input_repre)
                    input_repre = output_repre
            with _tf.name_scope(self.transpose_back_ns):
                output_repre = _tf.transpose(output_repre,[0,3,1,2])
        return output_repre
    def get_l2_loss(self):
        with _tf.name_scope(self.name_scope):
            l2_loss = _tf.add_n([conv.get_l2_loss() for conv in self.conv_layers])
        return l2_loss

class CombineSubgraphs(object):
    def __init__(self,input_size,output_size,layer_num,norm_flag=True,dropout_flag=False,res_flag=True\
                 ,activation_func=_tf.nn.leaky_relu,summ_flag=False,name_scope='CombineSubgraphs'):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_num = layer_num
        self.norm_flag = norm_flag
        self.dropout_flag = dropout_flag
        self.res_flag = res_flag
        self.activation_func = activation_func
        self.summ_flag = summ_flag
        with _tf.name_scope(name_scope) as self.name_scope:
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
        with _tf.name_scope(self.name_scope):
            with _tf.name_scope('Transpose') as self.transpose_ns:
                pass
            with _tf.name_scope('Layer%d'%0) as tmp_name_scope:
                self.name_scope_layers.append(tmp_name_scope)
                self.dense_layers.append(Dense(input_size*2,output_size,activation_func=linear_activation,summ_flag=self.summ_flag))
                self.norm_layers.append(LayerNorm([output_size,],summ_flag=summ_flag))
            for ln in range(1,layer_num):
                with _tf.name_scope('Layer%d'%ln) as tmp_name_scope:
                    self.name_scope_layers.append(tmp_name_scope)
                    self.dense_layers.append(Dense(output_size,output_size,activation_func=linear_activation,summ_flag=self.summ_flag))
                    self.norm_layers.append(LayerNorm([output_size,],summ_flag=summ_flag))
            with _tf.name_scope('TransposeBack') as self.transpose_back_ns:
                pass
    def __call__(self,subgraph_repres,train_flag):
        '''
        input arguments:
            subgraph_repres is the subgraphs' representations of type tf.Tensor and of shape [1,C,N-index+1,1]
            closer subgraphs in subgraph_repres share more common nodes.
        output: combined subgraph_repres of type tf.Tensor annd of shape [1,C',N-index,1]
        '''
        norm_flag = self.norm_flag
        dropout_flag = self.dropout_flag
        res_flag = self.res_flag
        activation_func = self.activation_func

        with _tf.name_scope(self.name_scope):
            with _tf.name_scope(self.transpose_ns):
                input_repre = _tf.transpose(subgraph_repres[0,:,:,0]) #[N-index+1,C]
                input_repre = _tf.concat([input_repre[:-1],input_repre[1:]],axis=-1) #[N-index+1,2C]
            for ns,dense,norm in zip(self.name_scope_layers,self.dense_layers,self.norm_layers):
                with _tf.name_scope(ns):
                    output_repre = dense(input_repre) #[N-index+1,C']
                    if norm_flag:
                        output_repre = norm(output_repre)
                    output_repre = activation_func(output_repre)
                    if dropout_flag:
                        output_repre = _tf.layers.dropout(output_repre,0.5,training=train_flag)
                    if res_flag and dense.input_size == dense.output_size:
                        output_repre = _tf.add(output_repre,input_repre)
                    input_repre = output_repre #[N-index+1,C']
            with _tf.name_scope(self.transpose_back_ns):
                output_repre = _tf.expand_dims(_tf.expand_dims(_tf.transpose(output_repre),axis=-1),axis=0)
        return output_repre
    def get_l2_loss(self):
        with _tf.name_scope(self.name_scope):
            l2_loss = _tf.add_n([dense.get_l2_loss() for dense in self.dense_layers])
        return l2_loss

class DiagUpdate(object):
    def __init__(self,input_size,output_size,norm_flag=True,summ_flag=True\
                 ,activation_func=_tf.nn.leaky_relu,name_scope='DiagUpdate'):
        self.input_size = input_size
        self.output_size =output_size
        self.norm_flag = norm_flag
        self.activation_func = activation_func
        self.summ_flag = summ_flag
        with _tf.name_scope(name_scope) as self.name_scope:
            pass
        self.build_model()
    def build_model(self,):
        with _tf.name_scope(self.name_scope):
            self.dense = Dense(self.input_size,self.output_size,activation_func=linear_activation,summ_flag=self.summ_flag)
            self.norm = LayerNorm([self.output_size],summ_flag=self.summ_flag)
    def get_output(self,E,index,train_flag):
        '''
        input arguments:
            E is edge attributes matrix of type tf.Tensor of shape [1,C,N,N]
            index is int, indexing the size of combining subgraphs
        output:
            combining gate of type tf.Tensor of shape [1,C',N-index,1]
        '''
        with _tf.name_scope(self.name_scope):
            diag_features = _tf.transpose(_tf.matrix_diag_part(E[0,:,index:,index:])) #[N-index,C]
            gates = self.dense(diag_features) #[N-index,C']
            if self.norm_flag:
                gates = self.norm(gates) #[N-index,C']
            gates = self.activation_func(gates) #[N-index,C']
            gates = _tf.expand_dims(_tf.expand_dims(_tf.transpose(gates),0),-1) #[1,C',N-index,1]
            return gates
    def __call__(self,E,index,train_flag):
        return self.get_output(E,index,train_flag)
    def get_l2_loss(self):
        with _tf.name_scope(self.name_scope):
            return self.dense.get_l2_loss()

class CombineGate(DiagUpdate):
    def __init__(self,input_size,output_size,norm_flag=True,summ_flag=True,name_scope='CombineGate'):
        super().__init__(input_size,output_size,norm_flag,summ_flag,activation_func=_tf.sigmoid,name_scope=name_scope)

class InitialSubgraph(DiagUpdate):
    def __call__(self,E,train_flag):
        return self.get_output(E,1,train_flag)

class IntegrateGraph(object):
    def __init__(self,input_size,output_size,layer_num,norm_flag=True,dropout_flag=False,res_flag=True\
                 ,activation_func=_tf.nn.leaky_relu,summ_flag=False,name_scope='IntegrateGraph'):
        self.summ_flag = summ_flag
        self.norm_flag = norm_flag
        with _tf.name_scope(name_scope) as self.name_scope:
            self.combiner = CombineSubgraphs(input_size,output_size,layer_num,norm_flag,dropout_flag,res_flag,\
                                        activation_func,summ_flag)
            self.gater = CombineGate(input_size,output_size,norm_flag,summ_flag)
    def __call__(self,subgraph_repres,E,index,train_flag):
        '''
        input arguments:
            subgraph_repres is the subgraphs' representations of type tf.Tensor and of shape [1,C,N-index+1,1]
                closer subgraphs in subgraph_repres share more common nodes.
            E is edge attributes matrix of type tf.Tensor of shape [1,C,N,N]
            index is int, indexing the size of combining subgraphs
        output:
            combining gate of type tf.Tensor of shape [1,C',N-index,1]
        '''
        with _tf.name_scope(self.name_scope):
            repres = self.combiner(subgraph_repres,train_flag) #[1,C',N-index,1]
            gates = self.gater(E,index,train_flag) #[1,C',N-index,1]
            return repres * gates
    def get_l2_loss(self,):
        with _tf.name_scope(self.name_scope):
            return _tf.add_n([self.combiner.get_l2_loss(),self.gater.get_l2_loss()])

class InitialSubgraph_v2(object):
    def __init__(self,input_size,output_size,norm_flag=True,summ_flag=True,name_scope='DiagUpdate'):
        self.input_size = input_size
        self.output_size =output_size
        self.norm_flag = norm_flag
        self.summ_flag = summ_flag
        with _tf.name_scope(name_scope) as self.name_scope:
            pass
        self.build_model()
    def build_model(self,):
        with _tf.name_scope(self.name_scope):
            self.dense = Dense(self.input_size,self.output_size,activation_func=linear_activation,summ_flag=self.summ_flag)
            self.norm = LayerNorm([self.output_size],summ_flag=self.summ_flag)
    def get_output(self,E,train_flag):
        '''
        input arguments:
            E is edge attributes matrix of type tf.Tensor of shape [1,C,N,N]
        output:
            combining gate of type tf.Tensor of shape [N-index,C',]
        '''
        index = 1
        with _tf.name_scope(self.name_scope):
            diag_features = _tf.transpose(_tf.matrix_diag_part(E[0,:,index:,index:])) #[N-index,C]
            gates = self.dense(diag_features) #[N-index,C']
            if self.norm_flag:
                gates = self.norm(gates) #[N-index,C']
            gates = _tf.sigmoid(gates) #[N-index,C']
            return gates
    def __call__(self,E,train_flag):
        return self.get_output(E,train_flag)
    def get_l2_loss(self):
        with _tf.name_scope(self.name_scope):
            return self.dense.get_l2_loss()

class CombineSubgraphs_v2(object):
    def __init__(self,input_size,output_size,layer_num,norm_flag=True,dropout_flag=False,res_flag=True\
                 ,activation_func=_tf.nn.leaky_relu,summ_flag=False,name_scope='CombineSubgraphs'):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_num = layer_num
        self.norm_flag = norm_flag
        self.dropout_flag = dropout_flag
        self.res_flag = res_flag
        self.activation_func = activation_func
        self.summ_flag = summ_flag
        with _tf.name_scope(name_scope) as self.name_scope:
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
        with _tf.name_scope(self.name_scope):
            with _tf.name_scope('Transpose') as self.transpose_ns:
                pass
            with _tf.name_scope('Layer%d'%0) as tmp_name_scope:
                self.name_scope_layers.append(tmp_name_scope)
                self.dense_layers.append(Dense(input_size*2,output_size,activation_func=linear_activation,summ_flag=self.summ_flag))
                self.norm_layers.append(LayerNorm([output_size,],summ_flag=summ_flag))
            for ln in range(1,layer_num):
                with _tf.name_scope('Layer%d'%ln) as tmp_name_scope:
                    self.name_scope_layers.append(tmp_name_scope)
                    self.dense_layers.append(Dense(output_size,output_size,activation_func=linear_activation,summ_flag=self.summ_flag))
                    self.norm_layers.append(LayerNorm([output_size,],summ_flag=summ_flag))
            with _tf.name_scope('TransposeBack') as self.transpose_back_ns:
                pass
    def __call__(self,subgraph_repres,train_flag):
        '''
        input arguments:
            subgraph_repres is the subgraphs' representations of type tf.Tensor and of shape [N-index+1,C,]
            closer subgraphs in subgraph_repres share more common nodes.
        output: combined subgraph_repres of type tf.Tensor annd of shape [N-index,C',]
        '''
        norm_flag = self.norm_flag
        dropout_flag = self.dropout_flag
        res_flag = self.res_flag
        activation_func = self.activation_func

        with _tf.name_scope(self.name_scope):
            with _tf.name_scope(self.transpose_ns):
                input_repre = _tf.concat([subgraph_repres[:-1],subgraph_repres[1:]],axis=-1) #[N-index+1,2C]
            for ns,dense,norm in zip(self.name_scope_layers,self.dense_layers,self.norm_layers):
                with _tf.name_scope(ns):
                    output_repre = dense(input_repre) #[N-index+1,C']
                    if norm_flag:
                        output_repre = norm(output_repre)
                    output_repre = activation_func(output_repre)
                    if dropout_flag:
                        output_repre = _tf.layers.dropout(output_repre,0.5,training=train_flag)
                    if res_flag and dense.input_size == dense.output_size:
                        output_repre = _tf.add(output_repre,input_repre)
                    input_repre = output_repre #[N-index+1,C']
        return output_repre
    def get_l2_loss(self):
        with _tf.name_scope(self.name_scope):
            l2_loss = _tf.add_n([dense.get_l2_loss() for dense in self.dense_layers])
        return l2_loss

class ConvCombineGate(object):
    def __init__(self,input_size,output_size,norm_flag=True,summ_flag=True,name_scope='ConvCombineGate'):
        self.input_size = input_size
        self.output_size =output_size
        self.norm_flag = norm_flag
        self.summ_flag = summ_flag
        with _tf.name_scope(name_scope) as self.name_scope:
            pass
        self.build_model()
    def build_model(self,):
        with _tf.name_scope('Transpose') as self.trans_ns:
            pass
        with _tf.name_scope(self.name_scope):
            self.conv = ConvCell([1,1,self.input_size,self.output_size],data_format='NHWC',summ_flag=self.summ_flag)
            self.norm = LayerNorm([self.output_size],summ_flag=self.summ_flag)
        with _tf.name_scope('TransposeBack') as self.trans_back_ns:
            pass
    def get_output(self,E,train_flag):
        '''
        input arguments:
            E is edge attributes matrix of type tf.Tensor of shape [1,C,N,N]
            index is int, indexing the size of combining subgraphs
        output:
            combining gate of type tf.Tensor of shape [1,C',N,N]
        '''
        with _tf.name_scope(self.name_scope):
            with _tf.name_scope(self.trans_ns):
                E = _tf.transpose(E,[0,2,3,1]) #[1,N,N,C]
            gates = self.conv(E) #[1,N,N,C']
            if self.norm_flag:
                gates = self.norm(gates) #[1,N,N,C']
            gates = _tf.sigmoid(gates) #[1,N,N,C']
            with _tf.name_scope(self.trans_back_ns):
                gates = _tf.transpose(gates,[0,3,1,2]) #[1,C',N,N]
            return gates
    def __call__(self,E,train_flag):
        return self.get_output(E,train_flag)
    def get_l2_loss(self):
        with _tf.name_scope(self.name_scope):
            return self.conv.get_l2_loss()

class IntegrateGraph_v2(object):
    def __init__(self,input_size,output_size,layer_num,norm_flag=True,dropout_flag=False,res_flag=True\
                 ,activation_func=_tf.nn.leaky_relu,summ_flag=False,name_scope='IntegrateGraph'):
        self.summ_flag = summ_flag
        self.norm_flag = norm_flag
        with _tf.name_scope(name_scope) as self.name_scope:
            self.combiner = CombineSubgraphs_v2(input_size,output_size,layer_num,norm_flag,dropout_flag,res_flag,\
                                        activation_func,summ_flag)
            with _tf.name_scope('DiagGates') as self.gate_ns:
                pass
    def __call__(self,subgraph_repres,gates,index,train_flag):
        '''
        input arguments:
            subgraph_repres is the subgraphs' representations of type tf.Tensor and of shape [N-index+1,C,]
                closer subgraphs in subgraph_repres share more common nodes.
            gates is gate matrix of type tf.Tensor of shape [1,C',N,N]
            index is int, indexing the size of combining subgraphs
        output:
            combining gate of type tf.Tensor of shape [N-index,C',]
        '''
        with _tf.name_scope(self.name_scope):
            repres = self.combiner(subgraph_repres,train_flag) #[1,C',N-index,1]
            with _tf.name_scope(self.gate_ns):
                diag_gates = _tf.transpose(_tf.matrix_diag_part(gates[0,:,index:,index:])) #[N-index,C']
            return repres * diag_gates
    def get_l2_loss(self,):
        with _tf.name_scope(self.name_scope):
            return _tf.add_n([self.combiner.get_l2_loss(),])

