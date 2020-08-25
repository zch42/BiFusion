import torch
from torch import nn
from torch_geometric.nn import GATConv

from layer.bipartite_gat import BipartiteGAT


class BiFusionLayer(torch.nn.Module):
    def __init__(self, protein_num, drug_num, disease_num,
                 protein_feature_num, drug_feature_num, disease_feature_num, hidden_dim):
        super(BiFusionLayer, self).__init__()
        self.protein_num = protein_num
        self.drug_num = drug_num
        self.disease_num = disease_num
        self.drug_protein = BipartiteGAT(drug_feature_num, protein_feature_num, hidden_dim, heads=1, dropout=0.1,
                                         flow='source_to_target', attention_concat=False, multi_head_concat=False)
        self.disease_protein = BipartiteGAT(disease_feature_num, protein_feature_num, hidden_dim, heads=1, dropout=0.1,
                                            flow='source_to_target', attention_concat=False, multi_head_concat=False)

        self.ppi_gat = GATConv(2 * hidden_dim, hidden_dim, heads=1, dropout=0.1, concat=False)

        self.protein_drug = BipartiteGAT(drug_feature_num, hidden_dim, hidden_dim, heads=1, dropout=0.1,
                                         flow='target_to_source', attention_concat=False, multi_head_concat=False)
        self.protein_disease = BipartiteGAT(disease_feature_num, hidden_dim, hidden_dim, heads=1, dropout=0.1,
                                            flow='target_to_source', attention_concat=False, multi_head_concat=False)

        self.act = nn.ReLU()

    def forward(self, ppi, drug_protein, disease_protein, drug_feature, disease_feature, protein_feature):
        edge_drug_protein = drug_protein.edge_index
        protein_feature_from_drug = self.act(self.drug_protein((drug_feature, protein_feature),
                                                               edge_drug_protein,
                                                               size=[self.drug_num, self.protein_num]))

        edge_disease_protein = disease_protein.edge_index
        protein_feature_from_disease = self.act(self.disease_protein((disease_feature, protein_feature),
                                                                     edge_disease_protein,
                                                                     size=[self.disease_num, self.protein_num]))

        protein_feature = torch.cat([protein_feature_from_drug, protein_feature_from_disease], dim=1)
        protein_feature = self.act(self.ppi_gat(protein_feature, ppi.edge_index))

        drug_feature = self.act(self.protein_drug((drug_feature, protein_feature),
                                                  edge_drug_protein,
                                                  size=[self.drug_num, self.protein_num]))

        disease_feature = self.act(self.protein_disease((disease_feature, protein_feature),
                                                        edge_disease_protein,
                                                        size=[self.disease_num, self.protein_num]))

        return drug_feature, disease_feature, protein_feature
