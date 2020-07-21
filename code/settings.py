import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('input_view', 0, 'View No. informative view, ACM:0, DBLP:1')
flags.DEFINE_float('weight_decay', 0.0001, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('fea_decay', 0.5, 'feature decay.')
flags.DEFINE_float('weight_R', 0.001, 'Weight for R loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('attn_drop', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('ffd_drop', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 50, 'number of iterations.')
flags.DEFINE_integer('n_clusters', 3, 'predict label early stop.')
flags.DEFINE_float('kl_decay', 0.1, 'kl loss decay.')




'''
infor: number of clusters 
'''
infor = {'ACM':3, 'DBLP':4}


'''
We did not set any seed when we conducted the experiments described in the paper;
We set a seed here to steadily reveal better performance of ARGA
'''
#seed = 7
#np.random.seed(seed)
#tf.set_random_seed(seed)

def get_settings(dataname, model, task):
    if  dataname != 'ACM' and dataname != 'DBLP':
        print('error: wrong data set name')
    if model != 'arga_ae' and model != 'arga_vae':
        print('error: wrong model name')
    if task != 'clustering' and task != 'link_prediction':
        print('error: wrong task name')

    if task == 'clustering':
        iterations = FLAGS.iterations
        clustering_num = infor[dataname]
        re = {'data_name': dataname, 'iterations' : iterations, 'clustering_num' :clustering_num, 'model' : model}
    elif task == 'link_prediction':
        iterations = 4 * FLAGS.iterations
        re = {'data_name': dataname, 'iterations' : iterations,'model' : model}

    return re

