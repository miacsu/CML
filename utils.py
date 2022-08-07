import os
import csv
import numpy as np

from sklearn.linear_model import RidgeClassifier, Lasso
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp


def load_data(modal_name, data_folder, phenotype_path):
    '''
    load multimodal data from ADNI
    return: imaging features (raw), labels, non-image data
    '''

    cli_infor = np.loadtxt(phenotype_path, delimiter=',')
    labels = cli_infor[:, -1]
    phonetic_data = cli_infor[:, 0:-1]

    features = np.loadtxt(os.path.join(data_folder, modal_name + ".csv"), delimiter=',')

    return features, labels, phonetic_data


def data_split(features, y, n_folds, seed):
    # split data by k-fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_splits = list(skf.split(features, y))
    return cv_splits


def get_node_features(features, y, train_ind, node_ftr_dim=None, feature_select=False):
    '''
    preprocess node features for GCN
    '''
    if feature_select:
        features, support = feature_selection(features, y, train_ind, node_ftr_dim)
        node_ftr = preprocess_features(features)
        return node_ftr, support
    else:
        node_ftr = preprocess_features(features)
        return node_ftr


# 基于Ridge的特征选择
def feature_selection(matrix, labels, train_ind, num_feature):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        num_feature  : number of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x num_feature)
        selector.support_   : index of selected features in the original feature set
    """

    estimator = RidgeClassifier()
    # estimator = Lasso(alpha=0.005, max_iter=3000)
    # estimator = SVC(kernel='linear')
    selector = RFE(estimator=estimator, n_features_to_select=num_feature, step=100, verbose=0)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    # x_data = matrix

    print("Number of features selected %d" % x_data.shape[1])

    return x_data, selector.support_


def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]])
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]

        if l in ['AGE', 'MMSE', 'PTEDUCAT']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph


def get_feature_graph(features):
    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    feature_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))

    return feature_graph


def write_raw_score(f, preds, labels):
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def get_confusion_matrix(preds, labels):
    matrix = [[0, 0], [0, 0]]
    for index, pred in enumerate(preds):
        if np.amax(pred) == pred[0]:
            if labels[index] == 0:
                matrix[0][0] += 1
            if labels[index] == 1:
                matrix[0][1] += 1
        elif np.amax(pred) == pred[1]:
            if labels[index] == 0:
                matrix[1][0] += 1
            if labels[index] == 1:
                matrix[1][1] += 1
    return matrix


def matrix_sum(A, B):
    return [[A[0][0] + B[0][0], A[0][1] + B[0][1]],
            [A[1][0] + B[1][0], A[1][1] + B[1][1]]]


def get_acc(matrix):
    return float(matrix[0][0] + matrix[1][1]) / float(sum(matrix[0]) + sum(matrix[1]))


def get_sen(matrix):
    return float(matrix[1][1]) / float(matrix[0][1] + matrix[1][1])


def get_spe(matrix):
    return float(matrix[0][0]) / float(matrix[0][0] + matrix[1][0])


def accuracy(preds, labels):
    """
    Accuracy, auc with masking. Acc of the masked samples
    """
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)


def get_auc(preds, labels, is_logit=True):
    '''
    input: logits, labels
    '''
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out


def preprocess_features(features):
    """Row-normalize feature matrix """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features