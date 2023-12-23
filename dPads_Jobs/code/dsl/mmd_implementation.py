import torch
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
from . import config

root = "/"
BATCH_SIZE = 64
N_INP = 17
N_OUT = 17
N_GEN_EPOCHS = 1
KERNEL_TYPE = "multiscale"

# Class definition Generative with Maximum Mean Discrepancy (GMMD)
class IPM(nn.Module):
    def __init__(self, n_start, n_out):
        super(IPM, self).__init__()
        self.fc1 = nn.Linear(n_start, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, n_out)

    def forward(self, samples):
        x = torch.sigmoid(self.fc1(samples))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

phi = IPM(N_INP, N_OUT).to(device)
# phi_optimizer = optim.RMSprop(phi.parameters(), lr=0.01)

def train_one_step(samples0, samples1):
    samples0 = Variable(samples0).to(device)
    samples1 = Variable(samples1).to(device)
    samples0 = phi(samples0)
    samples1 = phi(samples1)
    loss = MMD(samples0, samples1, KERNEL_TYPE)
    if config.globalmmdloss:
        config.globalmmdloss += loss
    else:
        config.globalmmdloss = loss
    return loss
  
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
     
    return torch.mean(XX + YY - 2. * XY)

def train_ipm(batch):
    # training loop
    iterations = 0
    zero_indices = None
    one_indices = None
    zero_indices = torch.where(batch[:,0]==0.0, 1, 0) # np.where(batch.cpu().clone().detach()[:,0]==0.0)[0]
    one_indices = torch.where(batch[:,0]==1.0, 1, 0) # one_indices = np.where(batch.cpu().clone().detach()[:,0]==1.0)[0]
    
    min_indices = min(len(zero_indices), len(one_indices))
        
    zero_indices = np.random.choice(len(zero_indices), min_indices, replace=False) 
    one_indices = np.random.choice(len(one_indices), min_indices, replace=False)

    samples0 = torch.tensor(batch[zero_indices,:][:,1:], dtype=torch.float)
    samples1 = torch.tensor(batch[one_indices,:][:,1:], dtype=torch.float)

    samples0 = samples0.view(samples0.size()[0], -1)
    samples1 = samples1.view(samples1.size()[0], -1)

    train_one_step(samples0, samples1)
    
    return torch.concat((batch[:,0:1], phi(batch[:,1:])), dim=1)