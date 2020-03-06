from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
            print('self.vars', self.vars['weights'])
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        #x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        #self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        #x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)

        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, name, v = 0, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        with tf.variable_scope('view', reuse=tf.AUTO_REUSE):
            self.vars['view_weights'] = tf.get_variable(name=name+str(v), shape=[input_dim, input_dim], trainable=True)
            print('self.vars:', self.vars['view_weights'])
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
       
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        #v_inputs = tf.layers.dense(inputs, 32, activation = tf.nn.relu)
        x = tf.transpose(inputs)
        tmp = tf.matmul(inputs, self.vars['view_weights'])
        x = tf.matmul(tmp, x)
        #x = tf.reshape(x, [-1])
        outputs = self.act(x)
        '''
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        #tmp = tf.matmul(inputs, self.vars['view_weights'])
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        '''
        return outputs

class InnerProductDecoderSingle(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, name, v = 0, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoderSingle, self).__init__(**kwargs)
        with tf.variable_scope('view', reuse=tf.AUTO_REUSE):
            self.vars['view_weights'] = tf.get_variable(name=name+str(v), shape=[input_dim, input_dim], trainable=True)
            print('self.vars:', self.vars['view_weights'])
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
       
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        #v_inputs = tf.layers.dense(inputs, 32, activation = tf.nn.relu)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        #x = tf.reshape(x, [-1])
        outputs = self.act(x)
        '''
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        #tmp = tf.matmul(inputs, self.vars['view_weights'])
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        '''
        return outputs


class InnerProductDecoder_MLP(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, v = 0, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder_MLP, self).__init__(**kwargs)
        with tf.variable_scope('view', reuse=tf.AUTO_REUSE):
            self.vars['view_weights'] = tf.get_variable(name="weight_view_"+str(v), shape=[input_dim, input_dim])
            self.vars['view_bais'] = tf.get_variable(name="weight_bais_"+str(v), shape=[input_dim])

        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        '''
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        #v_inputs = tf.layers.dense(inputs, 32, activation = tf.nn.relu)
        x = tf.transpose(inputs)
        tmp = tf.matmul(inputs, self.vars['view_weights'])
        x = tf.matmul(tmp, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        '''
        
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        tmp = tf.matmul(inputs, self.vars['view_weights'])+ self.vars['view_bais']
        x = tf.matmul(tmp, tf.transpose(tmp))
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        
        return outputs
class FuzeMultiEmbedding(Layer):
    """Fuze Multi-view encoder to a share embedding."""
    def __init__(self, fuze_func='mean',input_dim=32, dropout=0., **kwargs):
        super(FuzeMultiEmbedding, self).__init__(**kwargs)
        self.dropout = dropout
        self.fuze_func = fuze_func
        self.input_dim = input_dim


    def _call(self, inputs):
        #inputs = tf.nn.dropout(inputs, 1-self.dropout)
        print('inputs', inputs)
        self.W = tf.constant(0)
        if self.fuze_func == 'nothing':
            outputs = inputs[0]
        elif self.fuze_func == 'sum':
            outputs = tf.constant(0.9) * inputs[0] + tf.constant(0.1) * inputs[1]
        elif self.fuze_func == 'mean':
            outputs = tf.reduce_mean(inputs, axis=0)
        elif self.fuze_func == 'weight':
            numView = len(inputs)
            W = []
            for v in range(numView):
                self.vars['weight_'+str(v)] = weight_variable_glorot(1, 1, name="weight_"+str(v))
                W.append(self.vars['weight_'+str(v)])
            W = W/sum(W)
            outputs = 0
            for v in range(numView):
                outputs = tf.pow(W[v], 5) * inputs[v]
        elif self.fuze_func == 'max_pooling':
            outputs = tf.reduce_max(inputs, axis=0)
        elif self.fuze_func == 'concat':
            outputs = tf.concat(inputs, axis=1)
        elif self.fuze_func == 'MLP':
            concat = tf.concat(inputs, axis=1)
            concat1 = tf.layers.dense(concat, 32, activation=tf.nn.relu)
            concat2 = tf.layers.dense(concat1, 32, activation=tf.nn.relu)
            outputs = concat2
        elif self.fuze_func == 'att':
            num = len(inputs)
            W = []
            for n in range(num):
                W.append(tf.exp(self.attention( "V"+str(n), inputs[n] )))
            W = W/sum(W)
            self.W = W
            #W1 = tf.exp(self.attention( "V"+'0', inputs[0] ))

            #W2 = tf.exp(self.attention( "V"+'1', inputs[1] ))
            outputs = 0
            for n in range(num):
                outputs += W[n] * inputs[n]
            #w1 = W1/(W1+W2)
            #w2 = W2/(W1+W2)

            #outputs = w1 * inputs[0] + w2 * inputs[1]
            #self.w1 = w1
            #self.w2 = w2
        elif self.fuze_func == 'sem_att':
            gamma = 0.5
            self.vars['p_weight'] = weight_variable_glorot(self.input_dim, 1, name="p_weight")
            num = len(inputs)
            W = []
            with tf.variable_scope('p', reuse = tf.AUTO_REUSE):
                for n in range(num):
                    trans = tf.layers.dense(inputs[n], self.input_dim, activation = tf.nn.tanh, name='trans')
                    wei = tf.matmul(trans, self.vars['p_weight'])
                    wei = tf.reduce_mean(wei)
                    W.append(wei)
                soft_wei = tf.nn.softmax(W)
            outputs = 0
            self.W = soft_wei
            for n in range(num):
                outputs += tf.pow(soft_wei[n], gamma) * inputs[n]
            
            
        elif self.fuze_func == 'multi_att':
            out = []
            heads = 4
            for i in range(heads):
                num = len(inputs)
                W = []
                for n in range(num):
                    W.append(tf.exp(self.attention( "V"+str(i)+'_'+str(n), inputs[n] )))
                W = W/sum(W)
                self.W = W
                output = 0
                for n in range(num):
                     output += W[n] * inputs[n]
                out.append(output)
            outputs = tf.concat(out, axis=1)
        elif self.fuze_func == 'att2':
            mean = tf.reduce_mean(inputs, axis=0)
            with tf.variable_scope(self.name + '_vars'):
                self.vars['att_weights'] = weight_variable_glorot(self.input_dim, self.input_dim, name="att_view")
            num = len(inputs)
            W = []
            for n in range(num):
                tmp = tf.matmul(inputs[n], self.vars['att_weights']) 
                d = tf.exp(tf.matmul(tf.expand_dims(tmp, 1), tf.expand_dims(mean, 2)))
                d = tf.reshape(d, [-1,1])
                print('*******************d', d.shape)
                W.append(d)
            W = W/sum(W)
            self.W = W
            outputs = 0
            for n in range(num):
                outputs += W[n] * inputs[n]
            self.W = W
        elif self.fuze_func == 'MVE':
            numView = len(inputs)
            weights = []
            for v in range(numView):
                self.vars['weight_'+str(v)] = weight_variable_glorot(numView * self.input_dim, 1, name="weight_"+str(v))
                con_vec = tf.concat(inputs, axis=1)
                weight = tf.matmul(con_vec, self.vars['weight_'+str(v)])
                weights.append(weight)
            concat_weight = tf.concat(weights, axis=1)
            softmax_weight = tf.nn.softmax(concat_weight)
            self.W = softmax_weight
            #softmax_weight = tf.pow(softmax_weight, 2)
            outputs = 0
            for v in range(numView):
                outputs += tf.reshape(softmax_weight[:,v], [-1, 1]) * inputs[v]
        elif self.fuze_func == 'DIME':
            numView = len(inputs)
            fuze_layer = 0
            for v in range(numView):
                self.vars['MLP_weights_'+str(v)] = tf.get_variable(name="MLP_weights_"+str(v), shape=[self.input_dim, self.input_dim])
                self.vars['MLP_bias_'+str(v)] = tf.get_variable(name="MLP_bias_"+str(v), shape=[self.input_dim])
                fuze_layer += tf.matmul(inputs[v], self.vars['MLP_weights_'+str(v)]) + self.vars['MLP_bias_'+str(v)]
            outputs= tf.nn.sigmoid(fuze_layer)
      
        return outputs

    def attention(self, name, input_vec):
        att_w1 = tf.get_variable( name+"att_w1", shape=(self.input_dim, 64 ), initializer=tf.contrib.layers.xavier_initializer())
        att_b1 = tf.get_variable( name+"att_b1", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
        att_w2 = tf.get_variable( name+"att_w2", shape=(64,1), initializer=tf.contrib.layers.xavier_initializer())
        att_b2 = tf.get_variable( name+"att_b2", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        net_1 = tf.nn.sigmoid( tf.matmul(input_vec, att_w1) + att_b1 )
        net_2 = tf.nn.sigmoid( tf.matmul(net_1, att_w2) + att_b2 )
        return net_2  

class ClusteringLayer(Layer):
    """Clustering layer."""
    def __init__(self, input_dim, n_clusters=3, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.vars['clusters'] = weight_variable_glorot(self.n_clusters, input_dim, name="cluster_weight")
        #self.vars['clusters'].assign(self.initial_weights)

    def _call(self, inputs):
        q = tf.constant(1.0) / (tf.constant(1.0) + tf.reduce_sum(tf.square(tf.expand_dims(inputs,
            axis=1) - self.vars['clusters']), axis = 2)/tf.constant(self.alpha))
        q = tf.pow(q, tf.constant((self.alpha + 1.0) / 2.0))
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        
        return q


