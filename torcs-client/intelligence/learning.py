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
        hidden, output = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

class LSTM(nn.Module):
    def __init__(self, n_features, hidden_dim, output_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes feature vectors as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(n_features, hidden_dim)

        # The linear layer that maps from hidden state space to command space
        self.hidden2command = nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # return Variable(torch.zeros(1, 1, self.hidden_dim))
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, features, n_timesteps):
        lstm_out, self.hidden = self.lstm(features.view(n_timesteps, 1, -1), self.hidden)
        command = self.hidden2command(lstm_out.view(n_timesteps, -1))
        return command

def train_lstm(N_TIMESTEPS = 5):
    # Load our data
    t_head, f_head, t, f = load_data()
    print(t_head,t)
    print(f_head,f)

    # Create LSTM
    HIDDEN_DIM = 16
    L_RATE = 0.001
    n_output = t.size()[1]
    n_features = f.size()[1]
    lstm = LSTM(n_features, HIDDEN_DIM, n_output)
    loss_function = torch.nn.MSELoss()
    loss_vec = []
    optimizer = torch.optim.SGD(lstm.parameters(), lr = L_RATE)
    print(lstm)
    for epoch in range(2):
        for i in range(N_TIMESTEPS,f.size()[0]+1):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            lstm.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            lstm.hidden = lstm.init_hidden()

            # Step 2. Prepare data (concat multiple time steps)
            features_timesteps = f[i-N_TIMESTEPS:i,:]

            # Step 3. Run our forward pass.
            command = lstm(features_timesteps, N_TIMESTEPS)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(command, t[i-N_TIMESTEPS:i])
            loss_vec.append(loss.data[0])
            if i % 100 == 0:
                print(epoch,i,loss.data[0])
            loss.backward()
            optimizer.step()
    return lstm, loss_vec

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

def train_rnn(n_timesteps_used = 2):
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
    print('Training RNN...')
    n_iters = 1000
    learning_rate = 0.0000001
    loss_vec = []
    for iter in range(1,n_iters+1):
        hidden = rnn.init_hidden()
        rnn.zero_grad()

        # Predict steer, acc and brake based on n_timesteps before
        ix = random_integers(n_timesteps_used+1, f.data.shape[0]-n_timesteps_used+1)
        for i in reversed(range(n_timesteps_used+1)):
            output, hidden = rnn(f[ix-i:ix-i+1,:], hidden)

        loss = loss_func(output, t[i])
        loss_vec.append(loss.data[0])
        loss.backward()
        if iter % 100 == 0:
            print(iter,loss.data[0])

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)
    return rnn,loss_vec

if __name__ == "__main__":
    # Train and test feed forward network
    if len(sys.argv)==1:
        print('Please provide an argument (ff or rnn)')
        quit()
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
        net,loss_vec = train_rnn()
        import matplotlib.pyplot as plt
        plt.plot(list(range(len(loss_vec))),loss_vec)
        plt.show()
    elif sys.argv[1] == 'lstm':
        print('Training LSTM...')
        net,loss_vec = train_lstm()
        import matplotlib.pyplot as plt
        plt.plot(list(range(len(loss_vec))),loss_vec)
        plt.show()
    else:
        print('Please provide an argument (ff or rnn)')
        quit()
