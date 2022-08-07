import torch
import torch.nn.functional as F


class concatconcatMLP(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, num_classes, dropout):
        super(concatconcatMLP, self).__init__()
        self.dropout = dropout
        self.relu = torch.nn.ReLU(inplace=True)

        input_dim = input_dim1 + input_dim2 + input_dim3
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
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

    def forward(self, x1, x2, x3):

        features_fusion = torch.cat((x1, x2, x3), dim=1)

        logit = self.cls(features_fusion)

        return F.softmax(logit, dim=-1)