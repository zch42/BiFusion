import numpy as np
import torch

from dataloader.data_loader import BiFusionDataset
from model.BiFusion import BiFusionNet
from utils.evaluation_metrics import auroc, auprc

hidden_dim_1 = 256
hidden_dim_2 = 128

batch_num = 512

global_protein_num = 13460
global_drug_num = 256
global_disease_num = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_val_test(phase, epoch, batch, model, ppi, drug_protein, disease_protein,
                   drug_feature, disease_feature, protein_feature, pair, gt):
    if phase == 'Train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    logging_info = {}
    logging_info.update({'%s Epoch' % phase: epoch, 'batch': batch})
    prob = model(ppi, drug_protein, disease_protein, drug_feature, disease_feature, protein_feature, pair)
    gt = gt.float().to(device)

    weight = class_weight[gt.long()].to(device)
    loss_func = torch.nn.BCELoss(weight=weight, reduction='mean').to(device)
    loss = loss_func(prob, gt)

    if phase == 'Train':
        loss.backward()
        optimizer.step()

    logging_info.update({'loss': '%.04f' % loss.data.item()})
    logging_info.update({'auroc': '%.04f' % auroc(prob, gt)})
    logging_info.update({'auprc': '%.04f' % auprc(prob, gt)})
    return logging_info


if __name__ == '__main__':

    class_weight = torch.Tensor([1, 1])

    # Load preprocessed example data
    database = BiFusionDataset()

    # ppi
    ppi = database.protein_protein
    # drug_protein interactions
    drug_protein = database.drug_protein
    # disease_protein interactions
    disease_protein = database.disease_protein
    # drug_disease interactions
    drug_disease = database.drug_disease

    pair, label = drug_disease.edge_index, drug_disease.edge_label

    train_index, val_index, test_index = drug_disease.train_index, drug_disease.val_index, drug_disease.test_index
    pair_train, pair_val, pair_test = pair[:, train_index], pair[:, val_index], pair[:, test_index]
    label_train, label_val, label_test = label[train_index], label[val_index], label[test_index]

    # we only use drugs/diseases in the training set to construct similarity features
    selected_drug = np.unique(pair_train[0])
    selected_disease = np.unique(pair_train[1])

    # similarity features of drugs and diseases
    drug_feature = np.load('./data/processed/drug_feature.npy')[:, selected_drug]
    drug_feature = torch.from_numpy(drug_feature).float().to(device)

    disease_feature = np.load('./data/processed/disease_feature.npy')[:, selected_disease]
    disease_feature = torch.from_numpy(disease_feature).float().to(device)

    protein_feature = torch.zeros(global_protein_num, hidden_dim_1).float().to(device)

    # load data and model to GPU
    model = BiFusionNet(hidden_dim_1, hidden_dim_2,
                        global_protein_num, global_drug_num, global_disease_num,
                        protein_feature_num=hidden_dim_1, drug_feature_num=len(selected_drug),
                        disease_feature_num=len(selected_disease)).to(device)

    ppi, drug_protein, disease_protein = ppi.to(device), drug_protein.to(device), disease_protein.to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    for epoch in range(200):
        for batch, idx in enumerate(torch.split(torch.randperm(len(train_index)), batch_num)):
            train_logging = train_val_test('Train', epoch, batch, model, ppi, drug_protein, disease_protein,
                                           drug_feature, disease_feature, protein_feature,
                                           pair=pair_train[:, idx],
                                           gt=label_train[idx])
            print(train_logging)

        for batch, idx in enumerate(torch.split(torch.randperm(len(val_index)), batch_num)):
            val_logging = train_val_test('Val', epoch, batch, model, ppi, drug_protein, disease_protein,
                                         drug_feature, disease_feature, protein_feature,
                                         pair=pair_val[:, idx],
                                         gt=label_val[idx])
            print(val_logging)

        for batch, idx in enumerate(torch.split(torch.randperm(len(test_index)), batch_num)):
            test_logging = train_val_test('Test', epoch, batch, model, ppi, drug_protein, disease_protein,
                                          drug_feature, disease_feature, protein_feature,
                                          pair=pair_test[:, idx],
                                          gt=label_test[idx])
            print(test_logging)

        scheduler.step()
