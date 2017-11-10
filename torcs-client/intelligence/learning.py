# from intelligence.load_data import load_data
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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
    return target_header, features_header, targets,features

def train_ff_network():
    # Load our data
    t_head, f_head, targets, features = load_data()
    targets = Variable(torch.from_numpy(targets)).float()
    features = Variable(torch.from_numpy(features)).float()
    print(t_head,targets)
    print(f_head,features)

    ## Define feed forward network
    class FeedForwardNet(nn.Module):
        def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
            super(Net, self).__init__()
            self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer 1
            self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer 2
            self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

        def forward(self, f):
            h1 = F.relu(self.hidden1(f))      # activation function for hidden layer 1
            h2 = F.relu(self.hidden2(h1))     # activation function for hidden layer 1
            o = self.predict(h2)               # linear output
            return o
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

# t_head,f_head,t,f = load_data()
# net = train_network()
# features = Variable(torch.from_numpy(f).float())
# pred = net(features).data.numpy()
# for i,p in enumerate(pred):
    # print('Target:', t[i])
    # print('Pred:',p)


def train_recurrent_network():
    # Load our data
    t_head, f_head, targets, features = load_data()
    t = Variable(torch.from_numpy(targets)).float()
    f = Variable(torch.from_numpy(features)).float()
    print(t_head,targets)
    print(f_head,features)

    ## Define Recurrent neural network architecture
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

            self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
            self.i2o = nn.Linear(input_size + hidden_size, output_size)
            # self.softmax = nn.LogSoftmax()

        def forward(self, input, hidden):
            combined = torch.cat((input, hidden), 1)
            hidden = self.i2h(combined)
            output = self.i2o(combined)
            # output = self.softmax(output)
            return output, hidden

        def init_hidden(self):
            return Variable(torch.zeros(1, self.hidden_size))

    # Instantiate net
    net = RNN(f.data.shape[1], 10, 3)
    hidden = net.init_hidden()
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

if __name__ == "__main__":
    train_recurrent_network()
