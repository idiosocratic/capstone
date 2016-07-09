# model-building state value iteration agent, using archetypes

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
        self.iteration = 0  # how many states have we seen 
        
        
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
                       
    
       
    def get_best_action(self, current_archetype):  # get best action given the archetype of our current states
    
        possible_transitions = [] # list of archetypes we can get to from here
    
        for transition in self.transition_memory:
        
            if transition[0] == current_archetype:
            
                possible_transitions.append(transition)
        
        best_value = 0 # initialize
        
        best_transition = None # initialize        
                                
        for transition in possible_transitions:
        
            value_of_transition = self.arch_value_memory[transition[2]]  # value of next archetype given action
            
            if value_of_transition > best_value:
            
                best_value = value_of_transition
                
                best_transition = transition
                
        assert not best_transition == None
                
        best_action = best_transition[1]  # get action from best transition 
        
        return best_action
        
                 
      
    def is_arch_in_sas_memory(self, current_archetype):
    
        for transition in self.transition_memory:
        
            if transition[0] == current_archetype:
            
                return True
        
        return False
        
        
                
    def should_we_exploit(self):
    
        if self.iteration > self.memory_b4_exploit:
        
            return True
            
        return False
    
    
    
    def decay_epsilon(self):
    
        self.epsilon *= self.epsilon_decay
    
    
    
    def get_state_archetype(self, current_state):   
    
        closest_arch = None
        
        closest_distance = 1e3  # initialize as large number
    
        for archetype in self.archetypes:
         
            raw_arch = archetypes[archetype]  #  get array of archtype
            
            dist = get_L2_distance(raw_arch, current_state)  
            
            if dist < closest_distance:
            
                closest_distance = dist
                
                closest_arch = archetype
                
        assert not closest_arch == None
        
        return closest_arch
                
    
    
    def get_L2_distance(self, state1, state2):
    
        return np.linalg.norm(state1-state2)
        
        
        
    def
    def
    


# gym env
env = gym.make('CartPole-v0')

wondering_gnome = value_iter_agent(env.action_space)

for i_episode in xrange(10):
    observation = env.reset()
    
    episode_archetypes = []  # archetypes seen this episodes
    
    for t in xrange(200):
        env.render()
        print observation
        
        old_state = observation  # retain old state for updates
        old_state = np.reshape(old_state,(4, 1))  # reshape into numpy array
        
        action = env.action_space.sample()  # initialize random action
        
        if wondering_gnome.should_we_exploit():
         
            random_fate = np.random.random()
            
            if random_fate > wondering_gnome.epsilon:  # e-greedy implementation 
            
                wondering_gnome.get_best_action()
        
        observation, reward, done, info = env.step(action)
        
        new_state = observation  
        new_state = np.reshape(new_state,(4, 1))  # reshape into numpy array
        
        print "Old state, action, new state, reward: "
        print old_state, action, new_state, reward
        print "Shape: "
        print observation.shape
        
        
        
        
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break
       