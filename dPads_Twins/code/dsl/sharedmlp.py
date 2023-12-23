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
N_INP = 21
N_OUT = 21
N_GEN_EPOCHS = 1

# Class definition Generative with Maximum Mean Discrepancy (GMMD)
class Shared(nn.Module):
    def __init__(self, n_start, n_out):
        super(Shared, self).__init__()
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

shared = Shared(N_INP, N_OUT).to(device)

def train_one_step(samples):
    samples = Variable(samples).to(device)
    samples = shared(samples)

def train_shared(batch):
    # training loop
    iterations = 0

    samples = torch.tensor(batch[:,1:], dtype=torch.float)
    samples = samples.view(samples.size()[0], -1)

    train_one_step(samples)
    
    return torch.concat((batch[:,0:1], shared(batch[:,1:])), dim=1)