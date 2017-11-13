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

## Define Recurrent neural network architecture
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

# Load training data
def load_data(fpath = 'f-speedway.csv'):
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

def train_ff_network():
    t_head, f_head, targets, features = load_data()
    print(t_head,f_head)
    net = FeedForwardNet(n_feature=22, n_hidden1=15, n_hidden2=10, n_output=3)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    # Train network
    for t in range(30):
        prediction = net(features)     # input x and predict based on x
        loss = loss_func(prediction, targets)     # must be (1. nn output, 2. target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    return net

def train_rnn():
    # Load our data
    t_head, f_head, t, f = load_data()
    print(t_head,t)
    print(f_head,f)

    # Instantiate recurrent net
    rnn = RNN(f.data.shape[1], 10, 3)
    hidden = rnn.init_hidden()
    loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=0.5)

    # Train network
    n_iters = 10000
    n_timesteps = 5
    for iter in range(1,n_iters+1):
        hidden = rnn.init_hidden()
        rnn.zero_grad()

        # Predict steer,acc and brake based on n_timesteps before
        ix = random_integers(n_timesteps, f.data.shape[0])
        for i in reversed(range(n_timesteps)):
            output, hidden = rnn(f[ix-i:ix-i+1,:], hidden)

        loss = loss_func(output, t[i])
        loss.backward()
        if iter % 100 == 0:
            print(loss)

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)
    return rnn

if __name__ == "__main__":
    # Train and test feed forward network
    if sys.argv[1] == 'ff':
        print('Training feed forward network...')
        t_head,f_head,t,f = load_data()
        net = train_ff_network()
        pred = net(f).data.numpy()
        for i,p in enumerate(pred):
            print('Target:', t.data.numpy()[i])
            print('Pred:',p)
    elif sys.argv[1] == 'rnn':
        print('Training recurrent neural network...')
        net = train_rnn()
