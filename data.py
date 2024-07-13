# data.py

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class CropDataset_train(Dataset):

    def __init__(self, x, y, p, q, s, t):
        x = np.transpose(x, (0,3,1,2))
        y = np.transpose(y, (0,3,1,2))

        x_1 = torch.Tensor(x)
        x_2 = torch.flip(x_1, dims=(3,))
        x_3 = torch.cat((x_1, x_2), 0)
        x_4 = torch.cat((x_1, x_2), 0)        
        x_ = torch.cat((x_3, x_4), 0)
        
        y_1 = torch.Tensor(y)
        y_2 = torch.Tensor(y) 
        y_3 = torch.cat((y_1, y_2), 0)
        y_4 = torch.flip(y_3, dims=(3,))       
        y_ = torch.cat((y_3, y_4), 0)

        p_1 = torch.Tensor(p)
        p_2 = torch.Tensor(p)
        p_3 = torch.cat((p_1, p_2), 0)
        p_4 = torch.cat((p_1, p_2), 0)
        p_ = torch.cat((p_3, p_4), 0)
        
        q_1 = torch.Tensor(q)
        q_2 = torch.Tensor(q)
        q_3 = torch.cat((q_1, q_2), 0)
        q_4 = torch.cat((q_1, q_2), 0)
        q_ = torch.cat((q_3, q_4), 0)
        
        s_1 = torch.Tensor(s)
        s_2 = torch.Tensor(s)
        s_3 = torch.cat((s_1, s_2), 0)
        s_4 = torch.cat((s_1, s_2), 0)
        s_ = torch.cat((s_3, s_4), 0)       
        
        t_1 = torch.Tensor(t)
        t_2 = torch.Tensor(t)
        t_3 = torch.cat((t_1, t_2), 0)
        t_4 = torch.cat((t_1, t_2), 0)
        t_ = torch.cat((t_3, t_4), 0)        

        self.X = x_
        self.Y = y_
        self.P = p_
        self.Q = q_
        self.S = s_
        self.T = t_

    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.P[idx], self.Q[idx], self.S[idx], self.T[idx]


class CropDataset(Dataset):

    def __init__(self, x, y, p, q, s, t):
        x = np.transpose(x, (0,3,1,2))
        y = np.transpose(y, (0,3,1,2))

        x_ = torch.Tensor(x)
        y_ = torch.tensor(y)
        p_ = torch.Tensor(p)
        q_ = torch.Tensor(q)
        s_ = torch.Tensor(s)
        t_ = torch.Tensor(t)

        self.X = x_
        self.Y = y_
        self.P = p_
        self.Q = q_
        self.S = s_
        self.T = t_

    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.P[idx], self.Q[idx], self.S[idx], self.T[idx]

