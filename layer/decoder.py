import torch
from torch import nn


class MlpDecoder(torch.nn.Module):
    """
    MLP decoder
    return drug-disease pair predictions
    """

    def __init__(self, input_dim):
        super(MlpDecoder, self).__init__()
        self.mlp_1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim * 2), int(input_dim)),
                                   nn.ReLU())
        self.mlp_2 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim), int(input_dim // 2)),
                                   nn.ReLU())
        self.mlp_3 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim // 2), 1),
                                   nn.Sigmoid())

    def forward(self, drug_feature, disease_feature):
        pair_feature = torch.cat([drug_feature, disease_feature], dim=1)
        embedding_1 = self.mlp_1(pair_feature)
        embedding_2 = self.mlp_2(embedding_1)
        outputs = self.mlp_3(embedding_2)
        return outputs
