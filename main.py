# main.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import CropDataset_train, CropDataset
from model import LSFuseNet
from train import train, evaluate

# Assuming CUDA is available, otherwise set to 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


landsat_inc = 9
sentinel_inc = 13
hidden_dim = 40
sl = timesteps
sls = len(timesteps_s)
sll = len(timesteps_l)
ln_enc = 64 * landsat_inc * sll
sn_enc = 64 * sentinel_inc * sls
num_heads = 4
out_dim = 1
epochs = 251
num_layers = 1

def main():
    # Load your data here
    train_data = CropDataset_train(x_train, y_train, p_train, q_train, s_train, t_train)
    eval_data = CropDataset(x_eval, y_eval, p_eval, q_eval, s_eval, t_eval)

    # Initialize data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=64, shuffle=False)

    # Initialize model
    model = LSFuseNet(landsat_inc, sentinel_inc, ln_enc, sn_enc, hidden_dim, sl, sll, sls, out_dim, num_layers, num_heads)
    model.to(device)

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    train(model, optimizer, train_loader, eval_loader, epochs)

if __name__ == "__main__":
    main()
