# permutation agent, using archetypes

import numpy as np
import gym
#import archetypes #functions: get_archetypes & rebalance_archetypes
import random


class value_iter_agent(object):

    def __init__(self, action_space):
    
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Hey, not our space!'
        
        
        #hyperparameters, some aren't that hyper
        self.epsilon = 0.73  # how much do we explore
        self.epsilon_decay = 0.99  # rate at which we decay epsilon
        self.number_of_archetypes = 8  # number of archetypes to use for our states
        self.memory_b4_arch_initialize = 200  # how much memory before exploiting
        self.max_memory = 3e4  # maximum length of states to store in memory
        self.iteration = 0  # how many states have we seen 
        self.have_archetypes = False  # bool, do we have archetypes yet
        self.foresight = 3  # how many actions do we plan ahead
        
        #memories
        self.highest_score = 0  # highest score received so far 
        self.highest_permutation = {}  # dictionary of the best archetype_actions discovered so far
        self.arch_action_memory = []  # list of archetype action dicts
        self.archetypes = {}  # dict of our archetypes, key is their index, value is their array representation 
        self.memory = []  # memory of states from which we form our archetypes    

    
    
    def initialize_arch_action_list(self):
        
        needed_number = 2**(self.number_of_archetypes)
        
        arch_action_dict = {}
        
        arch_action_dict_list = []
        
        while len(arch_action_dict_list) < needed_number:
        
            for arch_num in range(self.number_of_archetypes):
        
                arch_action_dict[arch_num] = random.choice([0,1])
        
            if not arch_action_dict in arch_action_dict_list:
            
                arch_action_dict_list.append(arch_action_dict)
                
        
        self.arch_action_memory = arch_action_dict_list
        
        self.highest_permutation = random.choice(arch_action_dict_list)



    def get_best_action(self, current_archetype):  # get best action given the archetype of our current states
        
        best_action = self.highest_permutation[current_archetype] 
        
        return best_action    
        
        
        
    def replace_highest_arch_action_dict_if_needed(self, arch_action_dict, episode_score):    
    
        if episode_score > self.highest_score:
            
            self.highest_score = episode_score
        
            self.highest_permutation = arch_action_dict
        
                 
    
    def get_archetypes(self, number_of_archs, some_memory):
    
        assert not number_of_archs == 0
        assert not len(some_memory) == 0
        
        archetypes = []

        for i in range(number_of_archs):
            
            archetypes.append(random.choice(some_memory))
        
        archetypes_dict = {}   
        
        for index, arch in enumerate(archetypes): 
        
            archetypes_dict[index] = arch
 
        
        return archetypes_dict 
      
    
    
    def rebalance_archetypes(self, old_archetypes, updated_memory):  
    
        number_of_archs = len(old_archetypes)
        
        archetype_child_lists = [] # list of lists for states belonging to each archetype
        
        for num in range(number_of_archs):
            
            archetype_child_lists.append([])  # initialize lists for each archetype
        
        new_archetypes = []  # list for rebalanced archetypes
        
        for state in updated_memory:
            
            closest_arch = None # initialize index as invalid
            
            closest_distance = 1e3 # initialize as large number
            
            for arch_key in old_archetypes:
                
                dist = np.linalg.norm(state - old_archetypes[arch_key])  # get L2 distance between current state,arch pair
                    
                if dist < closest_distance:
                        
                    closest_distance = dist
                        
                    closest_arch = arch_key
                
            assert not closest_arch == None        

            archetype_child_lists[closest_arch].append(state) # assign state to closest arch child list


        assert len(archetype_child_lists) == len(old_archetypes)


        for index, child_list in enumerate(archetype_child_lists):
            
            append = False
            
            if not len(child_list) == 0:
            
                new_archetype = sum(child_list)/float(len(child_list)) # get average of child_list            
                
                new_archetypes.append(new_archetype)
                
                append = True
                
            if len(child_list) == 0:    
            
                new_archetype = old_archetypes[index]  # this one has no children for updating
                
                new_archetypes.append(new_archetype)
            
                append = True
            
            assert append   
                
                
        assert len(new_archetypes) == len(old_archetypes)
        assert len(archetype_child_lists) == len(old_archetypes)    
        
        new_archetypes_dict = {}
        
        for index, arch in enumerate(new_archetypes):  # populate our dict 
        
            new_archetypes_dict[index] = arch

        
        assert len(old_archetypes) == len(new_archetypes_dict)
        
        return new_archetypes_dict
        
        
                
    def should_we_explore(self):
        
        random_fate = np.random.random()
        
        if random_fate > self.epsilon:  # epsilon is decaying, over time random_fate will always be greater
        
            return False
            
        return True
    
    
    
    def decay_epsilon(self):
    
        self.epsilon *= self.epsilon_decay
    
    
    
    def get_state_archetype(self, current_state):
    
        closest_arch = None
        
        closest_distance = 1e3  # initialize as large number
    
        for archetype in self.archetypes:
         
            raw_arch = self.archetypes[archetype]  #  get array of archetype
            
            dist = self.get_L2_distance(raw_arch, current_state)  
            
            if dist < closest_distance:
            
                closest_distance = dist
                
                closest_arch = archetype
                
        assert not closest_arch == None
        
        return closest_arch  # returning archetype key
                
    
    
    def get_L2_distance(self, state1, state2):
    
        return np.linalg.norm(state1-state2)

    
    

    


# gym env
env = gym.make('CartPole-v0')

wondering_gnome = value_iter_agent(env.action_space)

wondering_gnome.initialize_arch_action_list()

rewards_list = []

for i_episode in xrange(400):
    observation = env.reset()
    
    #episode_archetypes = []  # archetypes seen this episode
    
    episode_score = 0
    
    epi_arch_action_dict = wondering_gnome.highest_permutation  # initialize our action dict
    
    if wondering_gnome.should_we_explore():
    
        epi_arch_action_dict = random.choice(wondering_gnome.arch_action_memory)  # we're exploring
    
    
    for t in xrange(200):
        env.render()
        #print observation
        
        old_state = observation  # retain old state for updates
        old_state = np.reshape(old_state,(4, 1))  # reshape into numpy array
        
        action = env.action_space.sample()  # initialize random action
        
        if wondering_gnome.have_archetypes:
        
            old_archetype = wondering_gnome.get_state_archetype(old_state)  # get old state's arch
        
            action = epi_arch_action_dict[old_archetype]  # get action from epi_arch_action_dict
        
            wondering_gnome.decay_epsilon()  # decay our exploration rate
                        
        
        observation, reward, done, info = env.step(action)
        
        #new_state = observation
        #new_state = np.reshape(new_state,(4, 1))  # reshape into numpy array
        
        episode_score += reward
        
        
        if not len(wondering_gnome.memory) > wondering_gnome.max_memory: # have we reached our maximum memory
        
            wondering_gnome.memory.append(old_state)
        
        #print "Old state, action, new state, reward: "
        #print old_state, action, new_state, reward
        #print "Shape: "
        #print observation.shape
        
        wondering_gnome.iteration += 1
        
        if done:
            print "Episode " + str(i_episode) + " finished after {} timesteps".format(t+1)
            print "Epsilon: "
            print wondering_gnome.epsilon
            break
     
    rewards_list.append(episode_score)

    wondering_gnome.replace_highest_arch_action_dict_if_needed(epi_arch_action_dict,episode_score)
    
    if wondering_gnome.have_archetypes:
    
        if i_episode % 5 == 0:  # rebalance archetypes every so often
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(wondering_gnome.archetypes, wondering_gnome.memory) 
            
            print "#3"
            assert not len(wondering_gnome.archetypes) < wondering_gnome.number_of_archetypes

        
            
    if wondering_gnome.should_we_initialize_archs():
    
        some_archetypes = wondering_gnome.get_archetypes(wondering_gnome.number_of_archetypes, wondering_gnome.memory)
            
        print "#1"
        assert not len(some_archetypes) < wondering_gnome.number_of_archetypes
            
        wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(some_archetypes, wondering_gnome.memory)
    
        print "#2"
        assert not len(wondering_gnome.archetypes) < wondering_gnome.number_of_archetypes
    
        wondering_gnome.have_archetypes = True
            
        for index in range(wondering_gnome.number_of_archetypes):
            
            wondering_gnome.arch_value_memory[index] = 0  # initialize archetype values
    
        

             
             
print rewards_list             
    
print wondering_gnome.archetypes 

print wondering_gnome.arch_value_memory

print wondering_gnome.transition_memory

print np.average(rewards_list)
    
    
