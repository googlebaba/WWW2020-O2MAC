import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def label_mask(labels):
    num = labels.shape[1]
    label_mask = np.dot(labels, np.array(range(num)).reshape((num, 1))).reshape((labels.shape[0]))
    return label_mask

def load_data(dataset):
    # load the data: x, tx, allx, graph
    if dataset == "ACM":
        dataset_path = "../data/ACM3025.mat"
    elif dataset == "DBLP":
        dataset_path = "../data/DBLP4057_GAT_with_idx.mat"

    data = sio.loadmat(dataset_path)
        #rownetworks = np.array([(data['PLP'] - np.eye(N)).tolist()]) #, (data['PLP'] - np.eye(N)).tolist() , (data['PTP'] - np.eye(N)).tolist()])
    print(dataset)
    if dataset == "ACM":
        truelabels, truefeatures = data['label'], data['feature'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['PAP']).tolist(), (data['PLP']).tolist()])
    elif dataset == "DBLP":
        truelabels, truefeatures = data['label'], data['features'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['net_APA']).tolist(), (data['net_APCPA']).tolist(), (data['net_APTPA']).tolist()])
    numView = rownetworks.shape[0]
    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    return np.array(rownetworks), numView, truefeatures, truelabels, y_train, y_val, y_test, train_mask, val_mask, test_mask



