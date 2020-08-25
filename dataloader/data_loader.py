import torch
from torch_geometric.data import InMemoryDataset


class BiFusionDataset(InMemoryDataset):
    def __init__(self, root='./data'):
        super(BiFusionDataset, self).__init__(root)
        with open(self.processed_paths[0], 'rb') as f:
            self.protein_protein = torch.load(f)
        with open(self.processed_paths[1], 'rb') as f:
            self.drug_protein = torch.load(f)
        with open(self.processed_paths[2], 'rb') as f:
            self.disease_protein = torch.load(f)
        with open(self.processed_paths[3], 'rb') as f:
            self.drug_disease = torch.load(f)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # preprocessed example data
        return ['protein_protein.pt', 'drug_protein.pt', 'disease_protein.pt', 'drug_disease.pt']

    def download(self):
        pass

    def process(self):
        pass
