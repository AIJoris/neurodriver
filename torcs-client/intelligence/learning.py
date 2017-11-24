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
from sklearn.decomposition import PCA

## Define feed forward network
class FeedForwardNet(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output, use_tanh= None):
        super(FeedForwardNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer 1
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer 2
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer
        self.use_tanh = use_tanh

    def forward(self, f):
        h1 = F.relu(self.hidden1(f))      # activation function for hidden layer 1
        h2 = F.relu(self.hidden2(h1))     # activation function for hidden layer 1
        if self.use_tanh:
            o = F.tanh(self.predict(h2))
        elif not self.use_tanh:
            o = F.softmax(self.predict(h2))
        elif self.use_tanh is None:
            o = self.predict(h2)
        return o


class LSTM(nn.Module):
    #def __init__(self, n_features, hidden_dim, output_size):
    def __init__(self, n_features, hidden_dim, n_output, use_tanh= None):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh

        # The LSTM takes feature vectors as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(n_features, hidden_dim)


        # The linear layer that maps from hidden state space to command space
        self.hidden2command = nn.Linear(hidden_dim, n_output)
        # tanh layer only used if use_tanh
        #self.hidden2command_tanh = nn.Tanh(self.hidden2command(hidden_dim))
        # sigmoid
        #self.hidden2command_softmax = F.softmax(self.hidden2command(hidden_dim))


        self.hidden = self.init_hidden()

    def init_hidden(self):
        # return Variable(torch.zeros(1, 1, self.hidden_dim))
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, features, n_timesteps):
        lstm_out, self.hidden = self.lstm(features, self.hidden)

        if self.use_tanh:
                command = nn.Tanh(self.hidden2command(lstm_out))
        elif not self.use_tanh:
                command = F.softmax(self.hidden2command(lstm_out))
        elif self.use_tanh is None:
                command = self.hidden2command(lstm_out)
        return command

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


def train_lstm( n_output = 3, N_TIMESTEPS = 4, use_tanh=None):
    # Load our data
    t_head, f_head, t, f = load_data()
    print(t_head,t)
    print(f_head,f)

    # Create LSTM
    HIDDEN_DIM = 10
    L_RATE = 0.0001
    #n_output = 1#t.size()[1]
    n_features = f.size()[1]


    #lstm = LSTM(n_features, HIDDEN_DIM, n_output)


    #n_feature, hidden_dim, output_size, use_tanh= None
    lstm = LSTM(n_features, HIDDEN_DIM, n_output, use_tanh = use_tanh)

    #net_steer = LSTM(n_feature=22, n_hidden1=15, n_hidden2=8, n_output=1, use_tanh = True)
    #net_speed = LSTM(n_feature=22, n_hidden1=15, n_hidden2=8, n_output=2, use_tanh = False)


    loss_function = torch.nn.MSELoss()
    loss_vec = []
    optimizer = torch.optim.SGD(lstm.parameters(), lr = L_RATE)
    print(lstm)
    for epoch in range(1):
        for i in range(N_TIMESTEPS, 10):
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
            loss = loss_function(command, t[i-N_TIMESTEPS:i,0])
            loss_vec.append(loss.data[0])
            if i % 100 == 0:
                print(epoch,i,loss.data[0])
                print('c',command.data.numpy()[0][0][0], 't',t[i-1,0].data.numpy()[0])
            loss.backward()
            optimizer.step()
    return lstm, loss_vec

# Load training data
def load_data(f_scale = False, fpath = 'out.csv', pca=False):
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
    if pca:
        n_feature = 10
        pca = PCA(n_components=n_feature)
        pca.fit(features)
        features = pca.transform(features)
        print('Conserved variance ratio: ', np.sum(pca.explained_variance_ratio_))
        print('features shape after PCA: ', features.shape)
    targets = Variable(torch.from_numpy(targets)).float()
    features = Variable(torch.from_numpy(features)).float()
    if f_scale == True:
        features, min, max = rescale(features)
    return target_header, features_header, targets,features

def rescale(input):
    mi = input - torch.min(input, 0)[0]
    ma = torch.max(input,0)[0]
    return (input - torch.min(input, 0)[0]) / (torch.max(input,0)[0] - torch.min(input,0)[0]), mi, ma

def train_ff_network(scale = False, pca=False):
    t_head, f_head, targets, features = load_data(f_scale = scale, pca=pca)
    print(t_head,f_head)
    # net = FeedForwardNet(n_feature=22, n_hidden1=15, n_hidden2=8, n_output=1, use_tanh = None)
    n_feature = features.size()[1]
    net_steer = FeedForwardNet(n_feature=n_feature, n_hidden1=15, n_hidden2=8, n_output=1, use_tanh = True)
    net_speed = FeedForwardNet(n_feature=n_feature, n_hidden1=15, n_hidden2=8, n_output=2, use_tanh = False)
    # print(net)
    print('Rescaling features:', scale)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    optimizer_steer = torch.optim.SGD(net_steer.parameters(), lr=0.03)
    optimizer_speed = torch.optim.SGD(net_speed.parameters(), lr=0.02)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    loss_vec_speed, loss_vec_steer = [], []
    # loss_vec = []

    # Train network
    for t in range(1000):
        prediction_steer, prediction_speed = net_steer(features), net_speed(features)
        loss_steer, loss_speed = loss_func(prediction_steer, targets[:,-1:]), loss_func(prediction_speed, targets[:,:-1])
        loss_vec_speed.append(loss_speed.data[0]), loss_vec_steer.append(loss_steer.data[0])
        optimizer_steer.zero_grad(), optimizer_speed.zero_grad()
        loss_steer.backward(), loss_speed.backward()
        optimizer_steer.step(), optimizer_speed.step()
        # prediction = net(features)     # input x and predict based on x
        # loss = loss_func(prediction, targets)     # must be (1. nn output, 2. target)
        # loss_vec.append(loss.data[0])
        # optimizer.zero_grad()   # clear gradients for next train
        # loss.backward()         # backpropagation, compute gradients
        # optimizer.step()        # apply gradients
    return net_speed, net_steer, loss_vec_speed, loss_vec_steer

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
        print('Please provide an argument (ff or rnn or lstm)')
        quit()
    if sys.argv[1] == 'ff':
        print('Training feed forward network...')
        pca = False
        f_scale = True
        t_head,f_head,t,f = load_data(f_scale = f_scale, pca=pca)
        _, _, t_valid, f_valid = load_data(f_scale=f_scale, fpath='e-track-4-custom.csv',)
        t[:,-1], min, max = rescale(t[:,-1])
        # net = train_ff_network(scale = f_scale)
        # pred = net(f).data.numpy()
        net_speed, net_steer, loss_vec_speed, loss_vec_steer = train_ff_network(scale = f_scale, pca=pca)
        pred_speed = net_speed(f).data.numpy()
        pred_steer = net_steer(f).data.numpy()
        loss = torch.nn.MSELoss()
        for i,p in enumerate(pred_speed):
            if i % 10000 == 0:
                print('Target acc/brake:', t.data.numpy()[i,0:-1])
                print('Pred acc/brake:',pred_speed[i])
                print('Target steer:', t.data.numpy()[i,-1])
                print('Pred steer:',pred_steer[i,-1])
        # pred_speed_valid = net_speed(f_valid)
        # pred_steer_valid = net_steer(f_valid)
        # plt.plot(list(range(len(loss_vec))),loss_vec)
        # print('Final loss:',loss_vec[-1])
        # plt.show()
        plt.plot(list(range(len(loss_vec_speed))),loss_vec_speed)
        plt.title('speed')
        print('Final loss speed:',loss_vec_speed[-1])
        plt.show()
        plt.plot(list(range(len(loss_vec_steer))),loss_vec_steer)
        plt.title('steer')
        print('Final loss steer:',loss_vec_steer[-1])
        # print("acc/brake validation loss: ", loss(pred_speed_valid, t_valid[:,:-1]).data[0])
        # print("steer validation loss: ", loss(pred_steer_valid, t_valid[:,-1:]).data[0])
        plt.show()

    elif sys.argv[1] == 'rnn':
        print('Training recurrent neural network...')
        net,loss_vec = train_rnn()
        plt.plot(list(range(len(loss_vec))),loss_vec)
        plt.show()
    elif sys.argv[1] == 'lstm':
        print('Training LSTM...')
        net,loss_vec = train_lstm()
        plt.plot(list(range(len(loss_vec))),loss_vec)
        plt.show()
    else:
        print('Please provide an argument (ff or rnn)')
        quit()
