import torch
import torch_geometric as tg
import torch.nn.functional as F
import math
import numpy as np

from models.metaGraph import metaGraph
from utils import get_feature_graph


class Attention(torch.nn.Module):
    def __init__(self, in_size, dim_k, dim_v):
        super(Attention, self).__init__()

        self.query = torch.nn.Linear(in_size, dim_k)
        self.key = torch.nn.Linear(in_size, dim_k)
        self.value = torch.nn.Linear(in_size, dim_v)

        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        atten = torch.nn.Softmax(dim=0)(torch.matmul(Q.T, K) * self._norm_fact)

        output = torch.matmul(V, atten)
        return output


class concatMGLF(torch.nn.Module):
    def __init__(self, x1_dim, x2_dim, num_classes, dropout, edgenet_input_dim, hgc, lg):
        super(concatMGLF, self).__init__()
        hidden_dim_list = [hgc for i in range(lg)]
        self.dropout = dropout
        bias = False
        input_dim = x1_dim + x2_dim
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.GCN = torch.nn.ModuleList()
        for i in range(lg):
            hidden_dim = input_dim if i == 0 else hidden_dim_list[i-1]
            # sym是采用对称归一化
            self.GCN.append(tg.nn.ChebConv(hidden_dim, hidden_dim_list[i], K=3, normalization='sym', bias=bias))

        self.GC1 = tg.nn.ChebConv(input_dim, input_dim, K=3, normalization='sym', bias=bias)
        self.GC2 = tg.nn.ChebConv(input_dim, input_dim, K=3, normalization='sym', bias=bias)
        self.SGC = tg.nn.ChebConv(input_dim, input_dim, K=3, normalization='sym', bias=bias)

        self.meta_adj_net = metaGraph(input_dim=edgenet_input_dim//2, dropout=dropout)

        self.attention = Attention(3*input_dim, input_dim, input_dim)

        cls_input_dim = sum(hidden_dim_list)
        self.cls = torch.nn.Sequential(
                torch.nn.Linear(cls_input_dim, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(128),
                torch.nn.Linear(128, num_classes))

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def get_adj(self, node_feature, metadata):
        '''
        get adj for GCN
        '''
        # number of subjects
        n = node_feature.shape[0]
        num_edge = n * n
        pd_ftr_dim = metadata.shape[1]

        meta_adj_index = np.zeros([2, num_edge], dtype=np.int64)
        meta_adj_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)

        feature_graph = get_feature_graph(node_feature)
        feature_adj_index = np.zeros([2, num_edge], dtype=np.int64)
        feature_adj_weight = np.zeros(num_edge, dtype=np.float32)
        feature_sim_score = np.zeros(num_edge, dtype=np.float32)

        flatten_ind = 0

        for i in range(n):
            for j in range(n):
                feature_adj_index[:, flatten_ind] = [i, j]
                feature_adj_weight[flatten_ind] = feature_graph[i][j]
                feature_sim_score[flatten_ind] = feature_graph[i][j]

                meta_adj_index[:, flatten_ind] = [i, j]
                meta_adj_input[flatten_ind] = np.concatenate((metadata[i], metadata[j]))

                flatten_ind += 1

        adj_index = feature_adj_index
        adj_weight = feature_adj_weight

        feature_adj_keep_ind = np.where(feature_adj_weight > 0.8)[0]
        feature_adj_index = feature_adj_index[:, feature_adj_keep_ind]
        feature_adj_weight = feature_adj_weight[feature_adj_keep_ind]

        feature_adj_index = torch.tensor(feature_adj_index, dtype=torch.long)
        feature_adj_weight = torch.tensor(feature_adj_weight, dtype=torch.float32)

        # normalization
        meta_adj_input = (meta_adj_input - meta_adj_input.mean(axis=0)) / meta_adj_input.std(axis=0)

        meta_adj_index = torch.tensor(meta_adj_index, dtype=torch.long)
        meta_adj_input = torch.tensor(meta_adj_input, dtype=torch.float32)
        meta_adj_weight = torch.squeeze(self.meta_adj_net(meta_adj_input))

        adj_weight = adj_weight + meta_adj_weight.detach().numpy()

        meta_adj_keep_ind = np.where(meta_adj_weight.detach().numpy() > 0.8)[0]
        meta_adj_index = meta_adj_index[:, meta_adj_keep_ind]
        meta_adj_weight = meta_adj_weight[meta_adj_keep_ind]

        adj_keep_ind = np.where(adj_weight > 1.6)[0]
        adj_index = adj_index[:, adj_keep_ind]
        adj_weight = adj_weight[adj_keep_ind]

        adj_index = torch.tensor(adj_index, dtype=torch.long)
        adj_weight = torch.tensor(adj_weight, dtype=torch.float32)

        return feature_adj_index, feature_adj_weight, meta_adj_index, meta_adj_weight, adj_index, adj_weight

    def forward(self, x1, x2, x3):

        features_fusion = torch.cat((x1, x2), dim=1)

        # get_adj
        feature_adj_index, feature_adj_weight, meta_adj_index, meta_adj_weight, adj_index, adj_weight\
            = self.get_adj(features_fusion.detach().numpy(), x3)

        features = F.dropout(features_fusion, self.dropout, self.training)

        x1 = self.relu(self.GC1(features, feature_adj_index, feature_adj_weight))
        x2 = self.relu(self.GC2(features, meta_adj_index, meta_adj_weight))
        com1 = self.SGC(features, feature_adj_index, feature_adj_weight)  # shared GC
        com2 = self.SGC(features, feature_adj_index, feature_adj_weight)  # shared GC

        x = (com1 + com2) / 2
        x = torch.cat((x1, x2, x), dim=1)
        x = self.attention(x)

        x = self.relu(self.GCN[0](x, adj_index, adj_weight))
        x0 = x

        for i in range(1, self.lg):
            x = F.dropout(x, self.dropout, self.training)
            x = self.relu(self.GCN[i](x, adj_index, adj_weight))
            jk = torch.cat((x0, x), dim=1)
            x0 = jk
        logit = self.cls(jk)

        return F.softmax(logit, dim=-1)

