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
        self.fc1 = nn.Linear(20, 20)
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
num = [i for i in range(0, 10797)]
epoch_num = 400

for epoch in range(0,epoch_num):
    sys.stdout.flush()
    running_loss = 0.0
    random.shuffle(num)
    '''
    Use batch here
    '''
    for i in range(0,100):
        data_train = torch.tensor(train_x[i*100:i*100+100])
        data_train = data_train.double()
    
        target_train = torch.tensor(train_y[i*100:i*100+100])
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

data_train = torch.tensor(train_x)
output_train = model(data_train)
result_train = output_train.detach().numpy()
train_pre_label=[]
for i in range(10797):
    x = -1
    pos = 0
    for j in range(10):
        if(result_train[i][j]>x):
            x=result_train[i][j]
            pos=j
    train_pre_label.append(pos)
train_real_label = []
for i in range(10797):
    pos = 0
    for j in range(10):
        if(train_y[i][j]==1):
            pos=j
    train_real_label.append(pos)
train_num_true = 0.0
for i in range(10797):
    if(train_pre_label[i]==train_real_label[i]):
        train_num_true+=1
print('The accuracy of this model in train set is :%f\n' %(train_num_true/10797))



data_test = torch.tensor(test_x)
data_test = data_test.double()
    
target_test = torch.tensor(test_y)
target_test = target_test.double()
    
output_test = model(data_test)
result_test = output_test.detach().numpy()
test_pre_label=[]
for i in range(4849):
    x = -1
    pos = 0
    for j in range(10):
        if(result_test[i][j]>x):
            x=result_test[i][j]
            pos=j
    test_pre_label.append(pos)
test_real_label = []
for i in range(4849):
    pos = 0
    for j in range(10):
        if(test_y[i][j]==1):
            pos=j
    test_real_label.append(pos)
test_num_true = 0.0
for i in range(4849):
    if(test_pre_label[i]==test_real_label[i]):
        test_num_true+=1
print('The accuracy of this model in test set is :%f\n' %(test_num_true/4849))