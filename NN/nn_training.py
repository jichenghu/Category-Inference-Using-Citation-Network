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
train_label_file = 'dest_train.csv'
train_y = np.loadtxt(open(train_label_file, "r"), delimiter=",", skiprows=0)
train_x_file = 'data_train.csv'
train_x = np.loadtxt(open(train_x_file, "r"), delimiter=",", skiprows=0)

# Define the network
'''
The numbers of neuron and layers can be modified
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 20)
        self.fc4 = nn.Linear(20, 15)
        self.fc5 = nn.Linear(15, 10)
    
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = f.softmax(self.fc5(x), dim = 1)
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
num = [i for i in range(0, 10797)]
epoch_num = 100

for epoch in range(0,epoch_num):
    sys.stdout.flush()
    running_loss = 0.0
    random.shuffle(num)
    '''
    Use batch here
    '''
    data_train = torch.tensor(train_x[epoch*100:epoch*100+100])
    data_train = data_train.double()
    
    target_train = torch.tensor(train_y[epoch*100:epoch*100+100])
    target_train = target_train.double()
    
    output = net(data_train)
    optimizer.zero_grad()
    loss = criterion(output, target_train)
    loss.backward()
    optimizer.step()
    
    running_loss = loss
    print('The loss of this epoch is:%f\n' %(running_loss))
    
torch.save(net, 'model.pkl')

test_label_file = 'dest_test.csv'
test_y = np.loadtxt(open(test_label_file, "r"), delimiter=",", skiprows=0)
test_x_file = 'data_test.csv'
test_x = np.loadtxt(open(test_x_file, "r"), delimiter=",", skiprows=0)

model = torch.load('model.pkl')

data_test = torch.tensor(test_x)
data_test = data_test.double()
    
target_test = torch.tensor(test_y)
target_test = target_test.double()
    
output = model(data_test)
