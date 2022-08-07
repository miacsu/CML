import torch
import torch.nn.functional as F
from models.crossModalFeatureFusion import CMFF


class CMFFConcat(torch.nn.Module):
    def __init__(self, x1_dim, x2_dim, x3_dim, CMFF_input_dim, CMFF_output_dim, num_classes, dropout):
        super(CMFFConcat, self).__init__()
        self.dropout = dropout
        self.relu = torch.nn.ReLU(inplace=True)

        self.CMFF = CMFF(x1_dim, x2_dim, CMFF_input_dim, CMFF_output_dim, CMFF_output_dim)

        mlp_input_dim = CMFF_output_dim + x3_dim

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

    def forward(self, x1, x2, x3):

        features_fusion, att_MRI, att_SNP, mlp1_weight, mlp2_weight = self.CMFF(x1, x2)
        x = torch.cat((features_fusion, x3), dim=1)

        logit = self.cls(x)

        return F.softmax(logit, dim=-1)