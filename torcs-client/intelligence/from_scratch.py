# from intelligence.load_data import load_data
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.random import random_integers
import os
import sys
import matplotlib.pyplot as plt

## Define feed forward network
class FeedForwardNet(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(FeedForwardNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer 1
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer 2
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, f):
        h1 = F.relu(self.hidden1(f))      # activation function for hidden layer 1
        h2 = F.relu(self.hidden2(h1))     # activation function for hidden layer 1
        o = self.predict(h2)               # linear output
        return o

# Load training data
def load_data(fpath = 'out.csv'):
    # alpine-1
    if os.path.relpath(".","..") != 'intelligence':
        fpath = 'intelligence/'+fpath
    df = pd.read_csv(fpath)
    header = np.array(df.columns)
    data = np.array(df)
    targets = data[:,0:3]
    features = data[:,3:]
    target_header = header[0:3]
    features_header = header[3:]
    targets = Variable(torch.from_numpy(targets)).float()
    features = Variable(torch.from_numpy(features)).float()
    return target_header, features_header, targets,features

def normalize(input):
    return (input - torch.mean(input)) / torch.std(input)

def train_ff_network(features, targets, norm_inputs=False):
    net = FeedForwardNet(n_feature=22, n_hidden1=50, n_hidden2=10, n_output=1)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    loss_vec = []
    # Train network
    print("Training FF network...")
    for t in range(500):
        prediction = net(features)     # input x and predict based on x
        loss = loss_func(prediction, targets)     # must be (1. nn output, 2. target)
        loss_vec.append(loss.data[0])
        if t % 100 == 0:
            print(t,loss.data[0])
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    return net, loss_vec

if __name__ == "__main__":
    print('Training feed forward network...')
    t_head,f_head,t,f = load_data()
    net, loss_vec = train_ff_network(f, t[:,2:3])
    pred = net(f).data.numpy()
    print("pred shape: ", pred.shape)
    for i,p in enumerate(pred):
        if i % 10000 == 0:
            print('Target:', t.data.numpy()[i])
            print('Pred:',p)
    plt.plot(list(range(len(loss_vec))),loss_vec)
    plt.show()
