import tensorflow as tf
import numpy as np
from model import ARGA, ARVGA
from optimizer import OptimizerAE, OptimizerVAE
import scipy.sparse as sp
from input_data import load_data
import inspect
from preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges, construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adjs_in, numView):
    
    placeholders = {
        'features': tf.placeholder(tf.float32),
        'adjs': tf.placeholder(tf.float32),
        'adjs_orig': tf.placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'attn_drop': tf.placeholder_with_default(0., shape=()),
        'ffd_drop': tf.placeholder_with_default(0., shape=()),
        'pos_weights': tf.placeholder(tf.float32),
        'fea_pos_weights': tf.placeholder(tf.float32),
        'p': tf.placeholder(tf.float32),
        'norm':tf.placeholder(tf.float32),
    }

    return placeholders


def get_model(model_str, placeholders, numView, num_features, num_nodes, num_clusters):
    #model = None
    if model_str == 'arga_ae':
        model = ARGA(placeholders, numView, num_features, num_clusters)

    elif model_str == 'arga_vae':
        model = ARVGA(placeholders, num_features, num_nodes, features_nonzero)

    return model

    
def format_data(data_name):
    # Load data
    #adj, features, y_test, tx, ty, test_maks, true_labels = load_data(data_name)
    print("&&&&&&&&&&&&&&&&&",data_name)
    rownetworks, numView, features, truelabels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(data_name)
    adjs_orig = []
    for v in range(numView):
        adj_orig = rownetworks[v]
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        #adj_orig.eliminate_zeros()
        adjs_orig.append(adj_orig)
    adjs_label = rownetworks

    adjs_orig = np.array(adjs_orig)
    adjs = adjs_orig
    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adjs_norm = preprocess_graph(adjs)

    num_nodes = adjs[0].shape[0]

    features = features
    num_features = features.shape[1]
    #features_nonzero = features[1].shape[0]
    fea_pos_weights = float(features.shape[0] * features.shape[1] - features.sum()) / features.sum()
    pos_weights = []
    norms = []
    for v in range(numView):
        pos_weight = float(adjs[v].shape[0] * adjs[v].shape[0] - adjs[v].sum()) / adjs[v].sum()
        norm = adjs[v].shape[0] * adjs[v].shape[0] / float((adjs[v].shape[0] * adjs[v].shape[0] - adjs[v].sum()) * 2)
        pos_weights.append(pos_weight)
        norms.append(norm)
    true_labels = truelabels
    feas = {'adjs':adjs_norm, 'adjs_label':adjs_label, 'num_features':num_features, 'num_nodes':num_nodes, 'true_labels':true_labels, 'pos_weights':pos_weights, 'norms':np.array(norms), 'adjs_norm':adjs_norm, 'features':features, 'fea_pos_weights':fea_pos_weights, 'numView':numView}
    return feas


def get_optimizer(model_str, model, numView, placeholders, num_nodes):
    if model_str == 'arga_ae':
        opt = OptimizerAE(model=model, preds_fuze=model.reconstructions_fuze, preds=model.reconstructions,
                          labels=placeholders['adjs_orig'],
                          p = placeholders['p'],
                          numView=numView,
                          pos_weights=placeholders['pos_weights'],
                          fea_pos_weights=placeholders['fea_pos_weights'],
                          norm=placeholders['norm'])
    elif model_str == 'arga_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           d_real=d_real,
                           d_fake=discriminator.construct(model.embeddings, reuse=True))
    return opt 

def update_test(model, opt, sess, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm, attn_drop, ffd_drop):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['pos_weights']: pos_weights})

    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})


    feed_dict.update({placeholders['attn_drop']: attn_drop})
    feed_dict.update({placeholders['ffd_drop']: ffd_drop})



    #feed_dict.update({placeholders['dropout']: 0})
    '''
    for key in feed_dict.keys():
        print('key', key)
        print('value', feed_dict[key])
    '''
    emb_ind = sess.run(model.embeddings, feed_dict=feed_dict)
    return emb_ind

def warm_update_test(model, opt, sess, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm, attn_drop, ffd_drop):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['pos_weights']: pos_weights})

    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})


    feed_dict.update({placeholders['attn_drop']: attn_drop})
    feed_dict.update({placeholders['ffd_drop']: ffd_drop})

    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    return emb

def warm_update(model, opt, sess, num_view, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm, attn_drop, ffd_drop):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['attn_drop']: attn_drop})
    feed_dict.update({placeholders['ffd_drop']: ffd_drop})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})


    #z_real_dist = np.random.randn(adj[0].shape[0], FLAGS.hidden2)
    #feed_dict.update({placeholders['real_distribution']: z_real_dist})
    avg_cost = []
    for j in range(5):
        for num in range(num_view):
            _, reconstruct_loss1 = sess.run([opt.opt_op_list[num], opt.cost_list[num]], feed_dict=feed_dict)
            avg_cost.append(reconstruct_loss1)

    return avg_cost

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm, attn_drop, ffd_drop):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['attn_drop']: attn_drop})
    feed_dict.update({placeholders['ffd_drop']: ffd_drop})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})


    reconstruct_loss = 0
    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss = 0
    g_loss = 0
    avg_cost = reconstruct_loss

    return avg_cost

def compute_q(model, opt, sess, adj_norm, adj_label, features, placeholders, pos_weights, fea_pos_weights, norm, attn_drop, ffd_drop):
    # construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['attn_drop']: attn_drop})
    feed_dict.update({placeholders['ffd_drop']: ffd_drop})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})


    #feed_dict.update({placeholders['dropout']: 0})
    '''
    for key in feed_dict.keys():
        print('key', key)
        print('value', feed_dict[key])
    '''

    #feed_dict.update({placeholders['real_distribution']: z_real_dist})

    q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)

    return q


def update_kl(model, opt, sess, adj_norm, adj_label, features, p, placeholders, pos_weights, fea_pos_weights, norm, attn_drop, ffd_drop, idx, label):
    # construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['attn_drop']: attn_drop})
    feed_dict.update({placeholders['ffd_drop']: ffd_drop})
    feed_dict.update({placeholders['pos_weights']: pos_weights})
    feed_dict.update({placeholders['fea_pos_weights']: fea_pos_weights})
    feed_dict.update({placeholders['norm']: norm})

    feed_dict.update({placeholders['p']: p})


    #feed_dict.update({placeholders['dropout']: 0})
    '''
    for key in feed_dict.keys():
        print('key', key)
        print('value', feed_dict[key])
    '''

    #feed_dict.update({placeholders['real_distribution']: z_real_dist})
    for j in range(5):
        _, kl_loss = sess.run([opt.opt_op_kl, opt.cost_kl], feed_dict=feed_dict)
    '''
    vars_embed = sess.run(opt.grads_vars, feed_dict=feed_dict)
    norms = []
    for n in range(vars_embed[0][0].shape[0]):
        norms.append(np.linalg.norm(vars_embed[0][0][n]))
    cluster_layer_q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)
    y_pred = cluster_layer_q.argmax(1)
    idx_list = []
    for n in range(len(y_pred)):
        if y_pred[n]==idx:
            idx_list.append(n)
    norms = np.array(norms)
    norms_tmp = norms[idx_list]
    label = np.array(label)[idx_list]
    tmp_q = cluster_layer_q[idx_list][:, idx]
    print('idx', idx)
    fw = open('./norm_q.txt', 'w')
    for n in range(len(norms_tmp)):
        str1 = str(norms_tmp[n]) + ' ' + str(tmp_q[n]) + ' ' + str(label[n])
        fw.write(str1)
        fw.write('\n')
    fw.close()
    '''
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    avg_cost = kl_loss

    return emb,avg_cost

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
