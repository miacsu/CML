import torch
import torch_geometric as tg
import torch.nn.functional as F
import numpy as np
from utils import get_feature_graph


class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, hgc, lg):
        super(GCN, self).__init__()
        hidden_dim_list = [hgc for i in range(lg)]
        self.dropout = dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.GCN = torch.nn.ModuleList()
        for i in range(lg):
            hidden_dim = input_dim if i == 0 else hidden_dim_list[i-1]
            self.GCN.append(tg.nn.ChebConv(hidden_dim, hidden_dim_list[i], K=3, normalization='sym', bias=bias))

        cls_input_dim = sum(hidden_dim_list)
        self.cls = torch.nn.Sequential(
                torch.nn.Linear(cls_input_dim, 256),
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

    def get_adj(self, feature):
        '''
        get adj for GCN
        '''
        # number of subjects
        n, pd_ftr_dim = feature.shape[0], feature.shape[1]
        num_edge = n * n

        feature_graph = get_feature_graph(feature)
        feature_adj_index = np.zeros([2, num_edge], dtype=np.int64)
        feature_adj_weight = np.zeros(num_edge, dtype=np.float32)
        feature_sim_score = np.zeros(num_edge, dtype=np.float32)

        flatten_ind = 0

        for i in range(n):
            for j in range(n):
                feature_adj_index[:, flatten_ind] = [i, j]
                feature_adj_weight[flatten_ind] = feature_graph[i][j]
                feature_sim_score[flatten_ind] = feature_graph[i][j]

                flatten_ind += 1

        feature_adj_keep_ind = np.where(feature_adj_weight > 0.9)[0]
        feature_adj_index = feature_adj_index[:, feature_adj_keep_ind]
        feature_adj_weight = feature_adj_weight[feature_adj_keep_ind]

        feature_adj_index = torch.tensor(feature_adj_index, dtype=torch.long)
        feature_adj_weight = torch.tensor(feature_adj_weight, dtype=torch.float32)

        return feature_adj_index, feature_adj_weight

    def forward(self, x):

        # get_adj
        adj_index, adj_weight = self.get_adj(x)

        x = self.relu(self.GCN[0](x, adj_index, adj_weight))
        x0 = x

        for i in range(1, self.lg):
            x = F.dropout(x, self.dropout, self.training)
            x = self.relu(self.GCN[i](x, adj_index, adj_weight))
            jk = torch.cat((x0, x), dim=1)
            x0 = jk
        logit = self.cls(jk)

        return F.softmax(logit, dim=-1)

