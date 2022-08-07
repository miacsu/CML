import torch
import math


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden, dropout=0.2):
        super(MLP, self).__init__()
        self.parser = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden, bias=True),
                )
        self.input_dim = input_dim
        self.model_init()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.parser(x)
        return output

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True


class CMFF(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden, dim_k, dim_v):
        super(CMFF, self).__init__()

        self.mlp1 = MLP(input_dim1, hidden)
        self.mlp2 = MLP(input_dim2, hidden)

        self.query1 = torch.nn.Linear(hidden, dim_k)
        self.key1 = torch.nn.Linear(hidden, dim_k)
        self.value1 = torch.nn.Linear(hidden, dim_v)

        self.query2 = torch.nn.Linear(hidden, dim_k)
        self.key2 = torch.nn.Linear(hidden, dim_k)
        self.value2 = torch.nn.Linear(hidden, dim_v)

        self._norm_fact = 1 / math.sqrt(dim_k)

        self.w = torch.nn.Linear(hidden, hidden)

    def forward(self, x1, x2):

        x1 = self.mlp1(x1)
        x2 = self.mlp2(x2)

        Q1 = self.query1(x1)
        K1 = self.key1(x1)
        V1 = self.value1(x1)

        Q2 = self.query2(x2)
        K2 = self.key2(x2)
        V2 = self.value2(x2)

        atten11 = torch.nn.Softmax(dim=0)(torch.matmul(Q1.T, K1) * self._norm_fact)
        atten12 = torch.nn.Softmax(dim=0)(torch.matmul(Q2.T, K1) * self._norm_fact)
        atten21 = torch.nn.Softmax(dim=0)(torch.matmul(Q1.T, K2) * self._norm_fact)
        atten22 = torch.nn.Softmax(dim=0)(torch.matmul(Q2.T, K2) * self._norm_fact)

        atten1 = atten11 + atten12
        atten2 = atten21 + atten22
        output1 = torch.matmul(V1, atten1)
        output2 = torch.matmul(V2, atten2)

        output = self.w(output1 + output2)

        return output, atten1, atten2, self.mlp1.parser[0].weight, self.mlp2.parser[0].weight