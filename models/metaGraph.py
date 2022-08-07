import torch


class metaGraph(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(metaGraph, self).__init__()
        hidden = 128
        self.parser = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden, bias=True),
                )
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = torch.nn.ReLU(inplace=True)
        self.elu = torch.nn.ReLU()

    def forward(self, x):
        x1 = x[:, 0:self.input_dim]
        x2 = x[:, self.input_dim:]
        h1 = self.parser(x1) 
        h2 = self.parser(x2) 
        p = (self.cos(h1, h2) + 1)*0.5
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True