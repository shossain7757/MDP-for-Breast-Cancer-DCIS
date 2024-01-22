#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 21:00:17 2022

@author: shossain
"""
import gym
import numpy as np
from collections import defaultdict



class BreastCancerDCISBandit(gym.Env):
    
    def __init__(self, config=None):
        
        '''
        I have 10 action space. Defined the action space from 0 to 9
        '''
        self.action_space = gym.spaces.Discrete(10)
        
        '''
        Feature space is 1 vector. Tumor Size
        
        '''
        
        self.observation_space = gym.spaces.Discrete(14)
        #self.observation_space = gym.spaces.Box(low=0, high=13, shape=(1, ), dtype=np.float64)
        
        '''
        I want each trajectory to start at higher tumor value with
        some randomisation.
        
        '''
        # initial state
        
        self.current_context = None
        self.log = ''
        
        # load matrices
        self.reward_matrix = np.load('reward_matrix2.npy')
        
        '''
        self.rewards_for_context = defaultdict(list)
        for key in range(0,len(reward_matrix)):
            for value in reward_matrix[key]:
                self.rewards_for_context[key].append(value)
                
        self.rewards_for_context = dict(self.rewards_for_context)
        '''
        
        
    def reset(self):
        self.current_context = 13
        return self.current_context
        
        
    def step(self, action):
        
        # log the chosen action
        
        self.log += f'Chosen action: {action}\n'
        
        # log the context
        
        self.log += f'Context: {self.current_context}\n'
        
        '''
        mean = self.rewards_for_context[self.current_context][action]
        stdev = [1,3]
        reward = np.random.normal(10, stdev[0]) if mean == 10 else np.random.normal(mean, stdev[1])
        '''
        
        reward = self.reward_matrix[self.current_context][action] * 1.0
        
        if (self.current_context == 0):
            done = True
        else:
            done = False
        
        present_context = self.current_context
        
        self.current_context -= 1
        
        return (
            np.array(present_context),
            reward,
            done,
            {"regret": 14 - reward}
        )   
    
    def render(self, mode=None):
        print(self.log)
        self.log = ''
    



