# model-building, q-learning agent using archetypes


import numpy as np
import gym
import archetypes #functions: get_archetypes & rebalance_archetypes


class value_iter_agent(object):

    def __init__(self, action_space):
    
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Hey, not our space!'
        
        
        #hyperparameters
        self.epsilon = 0.37  # how much do we explore
        self.epsilon_decay = 0.95  # rate at which we decay epsilon
        self.number_of_archetypes = 12  # number of archetypes to use for our states
        self.learning_rate = 0.23  # how quickly do we learn
        self.memory_b4_exploit = 200  # how much memory before exploiting 
        self.max_memory = 3e4  # maximum length of states to store in memory
        
        
        #memories
        self.transition_memory = []  # list for our SAS transitions 
        self.arch_value_memory = {}  # dict of pairs indicating the value of our archetypes
        self.archetypes = {}  # dict of our archetypes
        self.memory = []  # memory of states from which we form our archetypes    
    
    
    
    def add_to_sas_mem_if_needed(self, sas_tuple):
    
        if not sas_tuple[0] == sas_tuple[2]:  # is this a transition 
    
            if not sas_tuple in self.transition_memory:  # do we already know it
        
                self.transition_memory.append(sas_tuple)  
    
    
    
    def update_archetype_values(self, episode_archetypes, episode_score):
    
        score_weight = episode_score  # amount of weight to allocate 
        
        number_of_states = float(len(episode_archetypes))  # how many states in this episode
        
        represented_archetypes = set(episode_archetypes)
        
        arch_num_dict = {} # how many times did we see this archetype
        
        for arch in represented_archetypes:
        
            arch_episode_count = 0 
            
            for arch_state in episode_archetypes:
            
                if arch == arch_state:
                
                    arch_episode_count += 1
                    
            arch_num_dict[arch] = arch_episode_count 
        
        for key in arch_num_dict:
        
            arch_proportion = arch_num_dict[key] / float(number_of_states)  # how much does this arch represent our episode
            
            arch_weight = arch_proportion * score_weight
            
            current_value = self.arch_value_memory[key]
            
            # update value
            self.arch_value_memory[key] = (current_value * (1 - self.learning_rate)) + (arch_weight * self.learning_rate)
                       
    
       
    def
    def
    def
    def    