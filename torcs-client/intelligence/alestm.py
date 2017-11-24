# -*- coding: utf-8 -*-

# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from learning import load_data
import sys
import matplotlib.pyplot as plt


torch.manual_seed(1)

######################################################################
# Create the model:


class LSTMTagger(nn.Module):

    def __init__(self, n_features, hidden_dim, out_size, n_timesteps):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(n_features, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2command = nn.Linear(hidden_dim, out_size)
        self.hidden = self.init_hidden()
        self.n_timesteps = n_timesteps
        self.n_features = n_features

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input):
        output = input
        # for i in range(self.n_layers):
            # output, hidden = self.gru(output, hidden)
        lstm_out, self.hidden = self.lstm(
            input.view(self.n_timesteps, -1, self.n_features), self.hidden)
        hidden2 = F.relu(self.hidden2hidden(lstm_out.view(self.n_timesteps, -1)))
        output = self.hidden2command(hidden2)
        return output

######################################################################
# Train the model:

def prepare_input(features):
    tensor = torch.LongTensor(features)
    return autograd.Variable(tensor)

def train_lstm(n_timesteps=10):

    # Load our data
    t_head, f_head, t, f = load_data()
    print(t_head,t)
    print(f_head,f)

    # Create LSTM
    HIDDEN_DIM = 10
    L_RATE = 0.00000001
    N_TIMESTEPS = 150
    n_output = 3#t.size()[1]
    n_features = f.size()[1]
    model = LSTMTagger(n_features, HIDDEN_DIM, n_output, N_TIMESTEPS)
    loss_function = torch.nn.MSELoss()
    loss_vec = []
    optimizer = torch.optim.SGD(model.parameters(), lr = L_RATE)
    print(model)


    for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
        for i in range(N_TIMESTEPS, 10000):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Prepare data (concat multiple time steps)
            features_timesteps = f[i-N_TIMESTEPS:i,:]

            # Step 3. Run our forward pass.
            command = model(features_timesteps)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(command, t[i-N_TIMESTEPS:i])
            loss_vec.append(loss.data[0])
            if i % 100 == 0:
                print(epoch,i,loss.data[0])
                # print('c',command.data.numpy()[0][0][0], 't',t[i-1,0].data.numpy())
            loss.backward()
            optimizer.step()
    return model, loss_vec


if __name__ == "__main__":
    # Train and test feed forward network
    if len(sys.argv)==1:
        print('Please provide an argument (ff or rnn or lstm)')
        quit()
    if sys.argv[1] == 'ff':
        print('Training feed forward network...')
        t_head,f_head,t,f = load_data()
        net, loss_vec = train_ff_network()
        pred = net(f).data.numpy()
        for i,p in enumerate(pred):
            if i % 10000 == 0:
                print('Target:', t.data.numpy()[i])
                print('Pred:',p)
        plt.plot(list(range(len(loss_vec))),loss_vec)
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
