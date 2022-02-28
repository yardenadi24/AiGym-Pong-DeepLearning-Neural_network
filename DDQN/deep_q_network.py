import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self,lr,n_actions,name,input_dims,chkpt_dir):
        super(DeepQNetwork,self).__init__()
        #name of folder and file to save model
        self.cp_dir = chkpt_dir
        self.cp_file = os.path.join(self.cp_dir,name)

        #conv layers to transform the frames input into one layer of state
        self.conv1 = nn.Conv2d (input_dims[0], 32,8,stride = 4)
        self.conv2 = nn.Conv2d (32, 64,4,stride = 2)
        self.conv3 = nn.Conv2d (64, 64,3,stride = 1)

        #get the resulted dim form the transformation
        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        
        #fully connected
        self.fc1 = nn.Linear(fc_input_dims,512)
        self.fc2 = nn.Linear(512, n_actions)

        #optimize parameters for network
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)\
        
        #loss moodel set to mean squared error
        self.loss = nn.MSELoss()
        #device set to cpu
        self.device = T.device('cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self,in_dims):
        state = T.zeros(1,*in_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))
    
    #feed foeward the input state to get the values for each actions
    def forward(self,state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        
        conv_state = conv3.view(conv3.size()[0],-1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions


    def save_cp(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(),self.cp_file)
    
    def load_cp(self):
        print('... loading checkpoint...')
        self.load_state_dict(T.load(self.cp_file))
    
