import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from learning import load_data

class AutoEncoder(nn.Module):
    """docstring fo AutoEncoder."""
    def __init__(self, n_feature, n_hidden):
        super(AutoEncoder, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.output = nn.Linear(n_hidden, n_feature)

    def forward(self, input):
        x = F.sigmoid(self.hidden(input))
        return self.output(x)

net = AutoEncoder(n_feature=22, n_hidden=10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
print(net)

t_head, f_head, targets, features = load_data()
targets = Variable(torch.from_numpy(features)).float()
features = Variable(torch.from_numpy(features)).float()

# Train network
for t in range(30):
    prediction = net(features)     # input x and predict based on x
    loss = loss_func(prediction, targets)     # must be (1. nn output, 2. target)
    print("Loss at step {}: {}".format(t,loss))
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
