import torch
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
from . import config
from torch.nn import functional as F

root = "/"
BATCH_SIZE = 64
N_INP = 21
N_OUT = 21
N_GEN_EPOCHS = 1
KERNEL_TYPE = "multiscale"

# Class definition Generative with Maximum Mean Discrepancy (GMMD)
class Prop(nn.Module):
    def __init__(self, n_start, n_out):
        super(Prop, self).__init__()
        self.fc1 = nn.Linear(n_start, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, n_out)
        self.fc5 = nn.Linear(n_out, 1)

    def forward(self, samples, labels):
        x = torch.sigmoid(self.fc1(samples))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        y = self.fc5(x)
        loss = F.binary_cross_entropy_with_logits(labels, y)
        return x, loss


# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prop = Prop(N_INP, N_OUT).to(device)
# phi_optimizer = optim.RMSprop(phi.parameters(), lr=0.01)

def train_one_step(samples, labels):
    samples = Variable(samples).to(device)
    samples, loss = prop(samples, labels)
    if config.globalpropensityloss:
        config.globalpropensityloss += loss
    else:
        config.globalpropensityloss = loss
        
    return loss
  
def train_prop(batch):
    # training loop
    iterations = 0

    samples = torch.tensor(batch[:,1:], dtype=torch.float)
    samples = samples.view(samples.size()[0], -1)
    
    labels = torch.tensor(batch[:,0], dtype=torch.float)
    labels = labels.view(labels.size()[0], -1)

    train_one_step(samples, labels)
    
    return torch.concat((batch[:,0:1], prop(batch[:,1:], batch[:,0:1])[0]), dim=1)