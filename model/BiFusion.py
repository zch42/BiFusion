from layer.decoder import *
from layer.encoder import *


class BiFusionNet(torch.nn.Module):
    def __init__(self, hidden_dim_1, hidden_dim_2, protein_num, drug_num, disease_num,
                 protein_feature_num, drug_feature_num, disease_feature_num):
        super(BiFusionNet, self).__init__()
        self.encoder_1 = BiFusionLayer(protein_num, drug_num, disease_num, protein_feature_num, drug_feature_num,
                                       disease_feature_num, hidden_dim_1)
        self.encoder_2 = BiFusionLayer(protein_num, drug_num, disease_num, hidden_dim_1, hidden_dim_1, hidden_dim_1,
                                       hidden_dim_2)
        self.decoder = MlpDecoder(hidden_dim_2)

    def forward(self, ppi, drug_protein, disease_protein, drug_feature, disease_feature, protein_feature, pair):
        drug_feature, disease_feature, protein_feature = self.encoder_1(ppi, drug_protein, disease_protein,
                                                                        drug_feature, disease_feature, protein_feature)

        drug_feature, disease_feature, protein_feature = self.encoder_2(ppi, drug_protein, disease_protein,
                                                                        drug_feature, disease_feature, protein_feature)

        row, col = pair
        prediction = self.decoder(drug_feature[row, :], disease_feature[col, :]).flatten()
        return prediction
