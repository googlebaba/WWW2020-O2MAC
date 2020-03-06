import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def get_multi_sampled_index(adjs, size, center_num=3):
    numView = len(adjs)
    nodes = adjs[0].shape[0]
    indices = set(np.random.choice(nodes, center_num, False))
    indice = []
    for n in range(numView):
        indice.append(get_sampled_index(adjs[n], size, center_num, indices))
    indice_set = set(indice[0])
    for in_set in indice:
        indice_set = indice_set & set(in_set)
    return sorted(indice_set)


def get_sampled_index(adj, size, center_num, indices):
    n = adj.shape[0]
    pre_indices = set()
    #indices = set(np.random.choice(n, center_num, False))
    while len(indices) < size:
        if len(pre_indices) != len(indices):
            new_add = indices - pre_indices
            pre_indices = indices
            candidates = get_candidates(adj, new_add) - indices
        else:
            candidates = random_num(n, center_num, indices)
        sample_size = min(len(candidates), size-len(indices))
        if not sample_size:
            break
        if len(candidates) > size - len(indices):
            candidates = set(np.random.choice(list(candidates), size-len(indices), False))
        indices.update(candidates)
    return sorted(indices)


def get_candidates(adj, new_add):
    tmp = adj[sorted(new_add)].sum(axis=0)

    tmp1 = adj[sorted(new_add)].sum(axis=0).nonzero()[0]
    return set(adj[sorted(new_add)].sum(axis=0).nonzero()[0])


def random_num(n, num, indices):
    cans = set(np.arange(n)) - indices
    num = min(num, len(cans))
    if len(cans) == 0:
        return set()
    new_add = set(np.random.choice(list(cans), num, replace=False))
    return new_add

def preprocess_graph(adjs):
    numView = len(adjs)
    adjs_normarlized = []
    for v in range(numView):
        adj = sp.coo_matrix(adjs[v])
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray()
        adjs_normarlized.append(adj_normalized.tolist())
    return np.array(adjs_normarlized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adjs']: adj_normalized})
    feed_dict.update({placeholders['adjs_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


