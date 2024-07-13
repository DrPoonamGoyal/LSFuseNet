# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from model import LSFuseNet
from data import CropDataset_train, CropDataset

class MarginBasedContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(MarginBasedContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, target):
        distance = torch.nn.functional.pairwise_distance(anchor, positive)
        loss = torch.mean((1 - target) * torch.pow(distance, 2) +
                          (target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss

def train(model, optimizer, train_loader, eval_loader, epochs):
    criterion_pred = nn.MSELoss()
    criterion_triplet = MarginBasedContrastiveLoss(margin=0.5)

    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data_l, data_s, loc_l, loc_s, yr_l, target) in enumerate(train_loader):
            data_l, data_s, loc_l, loc_s, yr_l, target = (data_l.float()).to(device), (data_s.float()).to(device), (loc_l.float()).to(device), (loc_s.float()).to(device), (yr_l.float()).to(device), target.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            positive_pairs = []
            negative_pairs = []
            
            for i in range(len(target)):
                positive_idx, negative_idx = make_pairs(loc_l[i], loc_s[i], yr_l[i], loc_l, loc_s, yr_l)
                positive_pairs.append(positive_idx)
                negative_pairs.append(negative_idx)
                
            data_l_pos = data_l[positive_pairs]
            data_s_pos = data_s[positive_pairs]
            lnt_pos = data_l_pos
            snt_pos = data_s_pos
            
            anc_output, positive_out, negative_out = model(data_l, data_s, lnt_pos, snt_pos, data_l[negative_pairs], data_s[negative_pairs])
            target = target.repeat(anc_output.shape[1], 1).T
            output = anc_output.squeeze(-1)

            loss_1 = torch.sqrt(criterion_pred(output, target))
            loss_2 = criterion_triplet(anc_output, positive_out, target)
            loss_3 = criterion_triplet(anc_output, negative_out, target)
            loss = loss_1 + loss_2 + loss_3

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

        # Evaluate after every epoch
        evaluate(model, eval_loader)

def evaluate(model, eval_loader):
    criterion = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for data_l, data_s, loc_l, loc_s, yr_l, target in eval_loader:
            data_l, data_s, loc_l, loc_s, yr_l, target = (data_l.float()).to(device), (data_s.float()).to(device), (loc_l.float()).to(device), (loc_s.float()).to(device), (yr_l.float()).to(device), target.float().to(device)

            output = model(data_l, data_s)
            loss = criterion(output, target)
            total_loss += loss.item()

        avg_loss = total_loss / len(eval_loader)
        print(f"Validation Loss: {avg_loss}")

def make_pairs(loc_l, loc_s, yr_l, loc_l_all, loc_s_all, yr_l_all):
    positive_pairs = []
    negative_pairs = []

    for i in range(len(yr_l_all)):
        if loc_l == loc_l_all[i] and loc_s == loc_s_all[i] and yr_l == yr_l_all[i]:
            positive_pairs.append(i)
        else:
            negative_pairs.append(i)

    return positive_pairs, negative_pairs
