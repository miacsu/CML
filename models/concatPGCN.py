import torch
import torch_geometric as tg
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance


class concatPGCN(torch.nn.Module):
    def __init__(self, x1_dim, x2_dim, num_classes, dropout, hgc, lg):
        super(concatPGCN, self).__init__()
        hidden = [hgc for i in range(lg)]
        input_dim = x1_dim + x2_dim
        self.dropout = dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = torch.nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i-1]

            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K=3, normalization='sym', bias=bias))

        self.cls = torch.nn.Sequential(
                torch.nn.Linear(hidden[lg-1], 256),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(256),
                torch.nn.Linear(256, num_classes))

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def get_adj(self, features, metadata):
        n = features.shape[0]
        num_edge = n * n

        adj_index = np.zeros([2, num_edge], dtype=np.int64)
        adj_weight = np.zeros([num_edge, 1], dtype=np.float32)

        distv = distance.pdist(features, metric='correlation')
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))

        meta_distv = distance.pdist(metadata, metric='correlation')
        meta_dist = distance.squareform(meta_distv)
        meta_sigma = np.mean(meta_dist)
        meta_sim = np.exp(- meta_dist ** 2 / (2 * meta_sigma ** 2))

        adj = meta_sim + feature_sim

        flatten_ind = 0
        for i in range(n):
            for j in range(n):
                adj_index[:, flatten_ind] = [i, j]
                adj_weight[flatten_ind, :] = adj[i][j]
                flatten_ind += 1

        adj_keep_ind = np.where(adj_weight > 1.6)[0]
        adj_index = adj_index[:, adj_keep_ind]
        adj_weight = adj_weight[adj_keep_ind]

        return adj_index, adj_weight

    def forward(self, x1, x2, x3):

        features_fusion = torch.cat((x1, x2), dim=1)

        adj_index, adj_weight = self.get_adj(features_fusion.detach().numpy(), x3)

        adj_index = torch.tensor(adj_index, dtype=torch.long)
        adj_weight = torch.tensor(adj_weight, dtype=torch.float32)

        x = self.relu(self.gconv[0](features_fusion, adj_index, adj_weight))

        for i in range(1, self.lg):
            x = F.dropout(x, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, adj_index, adj_weight))
        logit = self.cls(x)

        return logit

