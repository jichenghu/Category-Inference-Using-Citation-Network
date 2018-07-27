import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
import sys
import random

# Prepare the input of the ensemble
'''
Get the path here and merge the input files
Caution: the input of every paper is 1-by-L vector, so the input matrix
have to be changed into reshape(num_paper, 1, L) to avoid mismatch
'''

# Prepare the goal of the ensemble
label_file = 'dest.csv'
y = np.loadtxt(open(label_file, "r"), delimiter=",", skiprows=0)
# Define the network
'''
The numbers of neuron and layers can be modified
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 10)
    
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.softmax(self.fc3(x), dim = 1)
        return x

'''
The weight initialize way can be modified
I already use the state-of-art way in most applications
'''
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.ReLU):
        nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))

# data_in is an one-dimensional array of ten elements
net = Net()
net.apply(weight_init)
net = net.double()

'''
The learning rate and the weight_deacy can be modified
Maybe take momentum into consideration?
'''
optimizer = optim.Adam(net.parameters(), lr = 1e-4, weight_decay = 1e-5)
criterion = nn.MSELoss()
'''
num is the array of paper id, random shuffle on every epoch
epoch_num can be modified
use batch-optimizing algorithm
'''
# num = [i for i in range(0, num_paper)]
epoch_num = 100

for epoch in range(0,epoch_num):
    sys.stdout.flush()
    running_loss = 0.0
    random.shuffle(num)
    '''
    Use barch here
    '''
    data_in = torch.tensor(data)
    data_in = data_in.double()
    
    target = torch.tensor(y)
    target = target.double()
    
    output = net(data_in)
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    running_loss = loss
    print('The loss of this epoch is:%f\n', running_loss)
    
torch.save(net, 'model.pkl')
