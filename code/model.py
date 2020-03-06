from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, InnerProductDecoderSingle, FuzeMultiEmbedding, ClusteringLayer
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass
def dense_to_sparse(a_t):
    #a_t = tf.constant(arr)
    idx = tf.where(tf.not_equal(a_t, 0))
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
    sparse = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), tf.shape(a_t, out_type=tf.int64))
    return sparse
class ARGA(Model):
    def __init__(self, placeholders, numView, num_features, num_clusters, **kwargs):
        super(ARGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.num_features = num_features
        #self.features_nonzero = features_nonzero
        self.adjs = placeholders['adjs']
        self.dropout = placeholders['dropout']
        self.attn_drop = placeholders['attn_drop']
        self.ffd_drop = placeholders['ffd_drop']
        self.num_clusters = num_clusters
        self.numView = numView
        self.build()

    def _build(self):

        with tf.variable_scope('Encoder', reuse=None):
            self.embeddings = []
            for v in range(self.numView):
                self.hidden1 = GraphConvolution(input_dim=self.num_features,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adjs[v],
                                                
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1_'+str(v))(self.inputs)
                                                  
                self.noise = gaussian_noise_layer(self.hidden1, 0.1)
                
                embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adjs[v],
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2_'+str(v))(self.noise)
                self.embeddings.append(embeddings)
            print('embeddings', self.embeddings)
            #Fuze layer


        self.cluster_layer = ClusteringLayer(input_dim=FLAGS.hidden2, n_clusters=self.num_clusters, name='clustering')
        self.cluster_layer_q = self.cluster_layer(self.embeddings[FLAGS.input_view])


        self.reconstructions = []
        for v in range(self.numView):
            view_reconstruction = InnerProductDecoder(input_dim=FLAGS.hidden2, name = 'e_weight_single_', v = v,
                                          act=lambda x: x,
                                          logging=self.logging)(self.embeddings[v])
            self.reconstructions.append(view_reconstruction)

        self.reconstructions_fuze = []
        for v in range(self.numView):
            view_reconstruction = InnerProductDecoder(input_dim=FLAGS.hidden2, name = 'e_weight_multi_', v = v,
                                          act=lambda x: x,
                                          logging=self.logging)(self.embeddings[FLAGS.input_view])
            self.reconstructions_fuze.append(view_reconstruction)
        # feature reconstruct


class ARVGA(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(ARVGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder'):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)


            self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2')(self.hidden1)

            self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging,
                                              name='e_dense_3')(self.hidden1)

            self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                          act=lambda x: x,
                                          logging=self.logging)(self.z)
            self.embeddings = self.z


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out
            
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise       
