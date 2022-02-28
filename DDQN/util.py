from tkinter import Frame
from cv2 import repeat
from matplotlib.axis import YAxis
import matplotlib.pyplot as plt
import numpy as np
import collections 
import cv2
import gym

def plot_learning_curve(x,scores,epsilons,filename):
    fig = plt.figure()
    ax = fig.add_subplot(111,label="1")
    ax2 = fig.add_subplot(111,label="2",frame_on=False)

    ax.plot(x,epsilons,color ="C0")
    ax.set_xlabel("Training steps", color="C0")
    ax.set_ylabel("Epsilons",color="C0")
    ax.tick_params(axis='x',color="C0")
    ax.tick_params(axis='y',color="C0")

    N = len(scores)

    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0,t-100):(t+1)])
    

    ax2.scatter(x,running_avg,color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score',color="C1")
    ax2.yaxis.set_label_poition('right')
    ax2.tick_parmas(axis='y',colors="C1")

    plt.savefig(filename)


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4,clip_reward = False, no_ops=0,fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2,self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first


    #overlode step of env for atari frame input
    def step(self,action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs,reward,done,info = self.env.step(action)

            if self.clip_reward:
                reward = np.clip(np.array([reward]),-1,1)[0]

            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0],self.frame_buffer[1])
        return max_frame , t_reward,done,info

    #overload reset of env for atari frame input
    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops>0 else 0

        for _ in range(no_ops):
            _,_,done,_ = self.env.step(0)
            if done:
                self.env.reset()

        if self.fire_first:
            assert self.env.unwrapped.get_Action_meanings()[1] == 'FIRE'
            obs,_,_,_ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs

        return obs   

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self,shape,env = None):
        super(PreprocessFrame,self).__init__(env)

        #openAi returns channels last and we want channels first for pytorch
        self.shape = (shape[2],shape[0],shape[1])

        #set obs space
        self.observation_space = gym.spaces.Box(low=0.0,high=1.0,shape = self.shape,dtype = np.float32)

    def observation(self,obs):

        #convert frame to gray scale
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        #resizing
        resized_screen = cv2.resize(new_frame,self.shape[1:],interpolation=cv2.INTER_AREA)

        #reshape and converting to array from cv2 object
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        #scaling
        new_obs = new_obs/ 255.0

        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self,env,repeat):
        super(StackFrames,self).__init__(env)
        self.observation_space = gym.spaces.Box(
                                env.observation_space.low.repeat(repeat,axis=0),
                                env.observation_space.high.repeat(repeat,axis=0),
                                dtype = np.float32
        )

        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self,observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)    

#build modified env
def make_env(env_name , shape = (84,48,1) ,repeat = 4, clip_reward = False, no_ops=0,fire_first=False):
    env  = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat,clip_reward = False, no_ops=0,fire_first=False)
    env = PreprocessFrame(shape,env)
    env = StackFrames(env,repeat)

    return env


