# from intelligence.load_data import load_data
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Fake data
# torch.manual_seed(1)
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
# x, y = Variable(x), Variable(y)
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

def load_data(fpath = 'intelligence/f-speedway.csv'):
    df = pd.read_csv(fpath)
    header = np.array(df.columns)
    data = np.array(df)
    targets = data[:,0:3]
    features = data[:,3:]
    target_header = header[0:3]
    features_header = header[3:]
    return target_header, features_header, targets,features


def train_network():
    # Load our data
    t_head, f_head, targets, features = load_data()
    targets = Variable(torch.from_numpy(targets)).float()
    features = Variable(torch.from_numpy(features)).float()
    print(t_head,targets)
    print(f_head,features)

    # Define network
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

        def forward(self, x):
            x = F.relu(self.hidden(x))      # activation function for hidden layer
            x = self.predict(x)             # linear output
            return x

    net = Net(n_feature=22, n_hidden=10, n_output=3)
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
