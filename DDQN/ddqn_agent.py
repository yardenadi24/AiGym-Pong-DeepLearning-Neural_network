import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DDQNAgent():
    def __init__(self, gamma, eps, lr,n_actions,in_dims,mem_size,batch_size,eps_min=0.01,eps_dec=5e-7,replace=1000,algo=None,env_name=None,chkpt_dir ='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = eps
        self.lr = lr 
        self.n_actions = n_actions
        self.input_dims = in_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_taget_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_Step_counter = 0

        self.memory = ReplayBuffer(mem_size,in_dims,n_actions)

        self.q_eval = DeepQNetwork(self.lr,self.n_actions,input_dims = self.input_dims,name = self.env_name+' '+self.algo+' '+'_q_eval',chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr,self.n_actions,input_dims = self.input_dims,name = self.env_name+' '+self.algo+' '+'_q_next',chkpt_dir=self.chkpt_dir)

    def choose_action(self,observation):
        if np.random.random() > self.epsilon:
            #if random greater then epsilon choose greedy action
                #feed state as tensor to the NN
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
                #forward pass in the layers to get the output that represent actions to choose
            actions = self.q_eval.forward(state)
                #choose arg max of those actions
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    

    def store_transition(self,state,action,reward,state_,done):
        self.memory.store_transition(state,action,reward,state_,done)
    
    def sample_memory(self):
        state,action,reward,new_state,done = self.memory.sample_buffer(self.batch_size)

        #convert to tensor and send to the device
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states,actions,rewards,states_,dones
    
    #replace target network
    def replace_target_network(self):
        if self.learn_Step_counter % self.replace_taget_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decerment_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save_cp()
        self.q_next.save_cp()
    
    def load_model(self):
        self.q_eval.load_cp()
        self.q_next.load_cp()
    

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        #zero the grad on the optimizer (initial)
        self.q_eval.optimizer.zero_grad()
        
        #replace target network if needed
        self.replace_target_network()

        #sample 
        states,actions,rewards,states_,dones = self.sample_memory()

        #fix for dim problem of forward output
        indices = np.arange(self.batch_size)

        #get predictions values form evaluation network
        q_pred = self.q_eval.forward(states)[indices,actions]


        #what are the valuse of the max actions for those states
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)


        max_actions = T.argmax(q_eval, dim=1)


        #if was done set the value to 0
        q_next[dones] = 0.0 

        #target calculation
        q_target = rewards + self.gamma*q_next[indices,max_actions]


        loss = self.q_eval.loss(q_target,q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_Step_counter+=1

        self.decerment_epsilon()

