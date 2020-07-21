from __future__ import division
from __future__ import print_function
from sklearn.cluster import KMeans
import settings
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')
import os
import numpy as np
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = 1
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
import tensorflow as tf
from metrics import clustering_metrics
from constructor import get_placeholder, get_model, compute_q, format_data, get_optimizer, warm_update, warm_update_test, update, update_test, update_kl
#from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn import preprocessing
import scipy.io as scio
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
def label_mask(labels):
    num = labels.shape[1]
    label_mask = np.dot(labels, np.array(range(num)).reshape((num, 1))).reshape((labels.shape[0]))
    return label_mask

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def count_num(labels):
    label_num = {}
    for label in labels:
        if label not in label_num:
            label_num[label] = 1
        else:
            label_num[label] += 1
    return label_num

def save_embed(emb, filename):
    fw = open(filename, 'w')
    for line in emb:
        fw.write(' '.join([str(s) for s in line]))
        fw.write('\n')
    fw.close()

loss = []
NMIs = []
class Clustering_Runner():
    def __init__(self, settings):

        print("Clustering on dataset: %s, model: %s, number of iteration: %3d" % (settings['data_name'], settings['model'], settings['iterations']))

        self.data_name = settings['data_name']
        self.iterations = 50
        self.kl_iterations = 0
        self.model = settings['model']
        self.n_clusters = settings['clustering_num']
        self.tol = 0.001
        self.time = 5
    
    def erun(self):
        tf.reset_default_graph()
        model_str = self.model
        # formatted data
        feas = format_data(self.data_name)
        placeholders = get_placeholder(feas['adjs'], feas['numView'])
        # construct model
        ae_model = get_model(model_str, placeholders, feas['numView'], feas['num_features'], feas['num_nodes'], self.n_clusters)
        # Optimizer
        opt = get_optimizer(model_str, ae_model, feas['numView'], placeholders, feas['num_nodes'])
        # Initialize session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True  #设置tf模式为按需赠长模式
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        # Train model
        
        pos_weights = feas['pos_weights']
        fea_pos_weights = feas['fea_pos_weights']

        for epoch in range(self.iterations):
            reconstruct_loss  = update(ae_model, opt, sess, feas['adjs'], feas['adjs_label'], feas['features'], placeholders, pos_weights, fea_pos_weights, feas['norms'], attn_drop=0., ffd_drop=0.)

            print('reconstruct_loss', reconstruct_loss)
          
            if (epoch+1) % 10 == 0:
                emb_ind = update_test(ae_model, opt, sess, feas['adjs'], feas['adjs_label'], feas['features'], placeholders,  pos_weights = pos_weights, fea_pos_weights = fea_pos_weights, norm = feas['norms'], attn_drop=0, ffd_drop=0)
                kmeans = KMeans(n_clusters=self.n_clusters).fit(emb_ind)
                print("PAP Epoch:", '%04d' % (epoch + 1))
                predict_labels = kmeans.predict(emb_ind)
                #print('emb1', emb_ind[1])
                label_num = count_num(predict_labels)
                print('view1 label_num:', label_num)
                cm = clustering_metrics(label_mask(feas['true_labels']), predict_labels)
                acc, f1_macro, precision_macro, nmi, adjscore,_ = cm.evaluationClusterModelFromLabel()
                NMIs.append(nmi)
                loss.append(reconstruct_loss)
        kmeans = KMeans(n_clusters=self.n_clusters).fit(emb_ind)
        y_pred_last = kmeans.labels_
        cm = clustering_metrics(label_mask(feas['true_labels']), y_pred_last)
        acc, f1_macro, precision_macro, nmi, adjscore, idx= cm.evaluationClusterModelFromLabel()
        init_cluster = tf.constant(kmeans.cluster_centers_)
        sess.run(
        tf.assign(ae_model.cluster_layer.vars['clusters'], init_cluster))
        q  = compute_q(ae_model, opt, sess, feas['adjs'], feas['adjs_label'], feas['features'], placeholders, pos_weights, fea_pos_weights, feas['norms'], attn_drop=0., ffd_drop=0.)
        p = target_distribution(q)
        for epoch in range(self.kl_iterations):
            emb, kl_loss = update_kl(ae_model, opt, sess, feas['adjs'], feas['adjs_label'], feas['features'], p, placeholders, pos_weights, fea_pos_weights, feas['norms'], attn_drop=0., ffd_drop=0., idx=idx, label = label_mask(feas['true_labels']))
            if epoch%10 == 0:
                kmeans = KMeans(n_clusters=self.n_clusters).fit(emb)
                predict_labels = kmeans.predict(emb)
                cm = clustering_metrics(label_mask(feas['true_labels']), predict_labels)
                acc, f1_macro, precision_macro, nmi, adjscore, _ = cm.evaluationClusterModelFromLabel()
                NMIs.append(nmi)
                loss.append(kl_loss)
            if epoch%5 == 0:
                q  = compute_q(ae_model, opt, sess, feas['adjs'], feas['adjs_label'], feas['features'], placeholders, pos_weights, fea_pos_weights, feas['norms'], attn_drop=0., ffd_drop=0.)
                p = target_distribution(q)
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                print('delta_label', delta_label)
                print("Epoch:", '%04d' % (epoch + 1))
                kmeans = KMeans(n_clusters=self.n_clusters).fit(emb)
                predict_labels = kmeans.predict(emb)
                cm = clustering_metrics(label_mask(feas['true_labels']), predict_labels)
                acc, f1_macro, precision_macro, nmi, adjscore, _ = cm.evaluationClusterModelFromLabel()
                if epoch > 0 and delta_label < self.tol:
                    print("early_stop")
                    break        
        print('NMI', NMIs)
        print('loss', loss)
        return acc, f1_macro, precision_macro, nmi, adjscore
        
if __name__ == '__main__':
    
    dataname = 'ACM'    # "ACM" or "DBLP"  
    #dataname = './data/DBLP4057_GAT_with_idx.mat'
    model = 'arga_ae'          # 'arga_ae' or 'arga_vae'
    task = 'clustering'         # 'clustering' or 'link_prediction'
    times = 10

    accs = []
    f1_macros = []
    precision_macros = []
    nmis = []
    adjscores = []
    settings = settings.get_settings(dataname, model, task)
    for t in range(times):
        print('times:%d'%t)
        if task == 'clustering':
            runner = Clustering_Runner(settings)
        acc, f1_macro, precision_macro, nmi, adjscore = runner.erun()
        accs.append(acc)
        f1_macros.append(f1_macro)
        precision_macros.append(precision_macro)
        nmis.append(nmi)
        adjscores.append(adjscore)

    acc_mean = np.mean(np.array(accs))    
    acc_std = np.std(np.array(accs), ddof=1)
    f1_mean = np.mean(np.array(f1_macros))    
    f1_std = np.std(np.array(f1_macros), ddof=1)
    precision_mean = np.mean(np.array(precision_macros))    
    precision_std = np.std(np.array(precision_macros), ddof=1)    
    nmi_mean = np.mean(np.array(nmis))    
    nmi_std = np.std(np.array(nmis), ddof=1)
    ari_mean = np.mean(np.array(adjscores))    
    ari_std = np.std(np.array(adjscores), ddof=1)

    print('ACC_mean=%f, ACC_std=%f, f1_mean=%f, f1_std=%f, precision_mean=%f, precision_std=%f, nmi_mean=%f, nmi_std=%f, ari_mean=%f, ari_std=%f' % (acc_mean, acc_std, f1_mean, f1_std, precision_mean, precision_std, nmi_mean, nmi_std, ari_mean, ari_std))

