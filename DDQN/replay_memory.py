import numpy as np

class ReplayBuffer():
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape) , dtype = np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_shape) , dtype = np.float32)
        self.action_mem = np.zeros(self.mem_size , dtype = np.int64)
        self.reward_mem = np.zeros(self.mem_size , dtype = np.float32)
        self.terminal_mem = np.zeros(self.mem_size , dtype = np.uint8)
    


    def store_transition(self,state, action,reward,state_,done):
        #index points to the next free space or the oldest memory that we can overwrite
        index = self.mem_cntr % self.mem_size 

        #store data at the index
        self.state_mem[index] = state
        self.new_state_mem[index] = state_ 
        self.action_mem[index] = action
        self.reward_mem[index] = reward  
        self.terminal_mem[index]  = done
        self.mem_cntr +=1
    
    
    def sample_buffer(self,batch_size):
        #get the amount of memeory stored
        max_mem = min(self.mem_cntr,self.mem_size)

        #randomly choose batch of memory
        batch = np.random.choice(max_mem,batch_size,replace=False)

        #assign memory to return
        states = self.state_mem[batch] 
        states_ =  self.new_state_mem[batch] 
        actions = self.action_mem[batch] 
        rewards = self.reward_mem[batch]    
        dones = self.terminal_mem[batch]  

        return states, actions,rewards,states_,dones
