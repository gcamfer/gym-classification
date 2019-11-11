import gym
from gym import spaces
import numpy as np

import logging
logger = logging.getLogger(__name__)


class Env4RLClassification(gym.Env):
    metadata = {'render.modes': ['none']}

    def __init__(self, X,y,batch_size=1,randomize=False,custom_rewards=None):
        super(Env4RLClassification, self).__init__()

        self.batch_size = batch_size
        self.randomize = randomize
        self.custom_rewards = custom_rewards
        self.episode_over = False
        
        if len(X) != len(y):
            print("ERROR incompatible sizes")
        
        # cast np y data
        y = np.array(y)
        actions = np.unique(y)
        # Action space
        self.action_space = spaces.Discrete(len(actions))
        
        # 1 epoch = 1 game
        #self.observation_space = spaces.Box(low=-1, high=1,shape=(self.env.getStateSize()))
        if self.randomize:
            new_indices = np.random.permutation(X.shape[0])
            np.take(X,new_indices,axis=0,out=X)
            np.take(y,new_indices,axis=0,out=y)
            
        self.X = X
        self.y = y
        
        self.current_indices = np.arange(batch_size)
        
    
    def _reset(self):
        self.episode_over = False
        self.true_labels = np.take(self.y,self.current_indices,axis=0)
        return np.take(self.X,self.current_indices,axis=0)
    
    
    def _step(self, action):

        # Update actions
        self.action = action
        # Take rewards for current actions
        reward = self._get_reward()
        # Update indices

        last_element = self.current_indices[-1]        
        if(max(self.current_indices) + self.batch_size) > len(self.X):
          self.episode_over = True
          ## greater
          if last_element == max(self.current_indices):
            self.current_indices += batch_size
            dif = max(self.current_indices)-len(self.X)
            self.current_indices[len(self.current_indices)-dif-1:len(self.current_indices)] = list(range(dif+1))
          else:
            self.current_indices = np.arange(last_element+1,last_element+1+self.batch_size,dtype=np.int32)
        else:
            self.current_indices += batch_size 

        
        # Update states for next step
        self.true_labels = np.take(self.y,self.current_indices,axis=0)
        self.status = np.take(self.X,self.current_indices,axis=0)
        
        return self.status, reward, self.episode_over, {}


    def _get_reward(self):
        return (self.action == self.true_labels)*1
