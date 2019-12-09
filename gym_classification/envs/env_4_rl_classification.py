import gym
from gym import spaces
import numpy as np

import logging
logger = logging.getLogger(__name__)


class Env4RLClassification(gym.Env):
    """
    Class definition of the Env4RLClassification class that allows to implement 
    in a simple way a trained classifier with reinforcement learning updates.
    
    Parameters
    ----------
    X : numpy matrix 
        Value for x features
    y : numpy array
        Target values
    """
    metadata = {'render.modes': ['none']}

    def __init__(self):
        super(Env4RLClassification, self).__init__()
        
    def init_dataset(self, X=None,y=None,batch_size=None,output_shape=None,randomize=False,custom_rewards=None):
        

        self.batch_size = batch_size
        self.output_shape = output_shape

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
        
        self.current_indices = np.arange(self.batch_size)
        
    
    def reset(self):
        self.episode_over = np.array([False]*len(self.current_indices))
        self.true_labels = np.take(self.y,self.current_indices,axis=0).ravel()
        if self.output_shape:
            return np.take(self.X,self.current_indices,axis=0).reshape(-1,*self.output_shape)
        else:
            return np.take(self.X,self.current_indices,axis=0)
        
    
    
    def step(self, action):

        # Update actions
        self.action = action
        # Take rewards for current actions
        reward = self._get_reward()
        # Update indices

        last_element = self.current_indices[-1]        
        if(max(self.current_indices) + self.batch_size) > len(self.X):
          self.episode_over = np.array([True]*len(self.current_indices))
          ## greater
          if last_element == max(self.current_indices):
            self.current_indices += self.batch_size
            dif = max(self.current_indices)-len(self.X)
            self.current_indices[len(self.current_indices)-dif-1:len(self.current_indices)] = list(range(dif+1))
          else:
            self.current_indices = np.arange(last_element+1,last_element+1+self.batch_size,dtype=np.int32)
        else:
            self.current_indices += self.batch_size 

        
        # Update states for next step
        self.true_labels = np.take(self.y,self.current_indices,axis=0).ravel()
        
        if self.output_shape:
            self.status = np.take(self.X,self.current_indices,axis=0).reshape(-1,*self.output_shape)
        else:
            self.status = np.take(self.X,self.current_indices,axis=0)
            
        
        
        return self.status, reward, self.episode_over, {}


    def _get_reward(self):
        return (self.action == self.true_labels)*1
        
    def seed(self):
        pass
