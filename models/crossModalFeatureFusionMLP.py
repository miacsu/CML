import torch
import torch.nn.functional as F
from models.crossModalFeatureFusion import CMFF


class CMFFMLP(torch.nn.Module):
    def __init__(self, x1_dim, x2_dim, CMFF_hidden, mlp_input_dim, num_classes, dropout):
        super(CMFFMLP, self).__init__()
        self.dropout = dropout
        self.relu = torch.nn.ReLU(inplace=True)

        self.CMFF = CMFF(x1_dim, x2_dim, CMFF_hidden, mlp_input_dim, mlp_input_dim)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_dim, 256),
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

    def forward(self, x1, x2):

        features_fusion, att_MRI, att_SNP, mlp1_weight, mlp2_weight = self.CMFF(x1, x2)

        logit = self.cls(features_fusion)

        return F.softmax(logit, dim=-1)