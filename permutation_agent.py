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
        self.learning_rate = 0.7  # how quickly do we learn
        #self.memory_b4_exploit = 200  # how much memory before exploiting 
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
        
        
        
    def update_high_score(self, episode_score):
    
        if episode_score > self.highest_score:
        
            self.highest_score = episode_score
    
    
    #def
    


# gym env
env = gym.make('CartPole-v0')

wondering_gnome = value_iter_agent(env.action_space)

rewards_list = []

for i_episode in xrange(400):
    observation = env.reset()
    
    episode_archetypes = []  # archetypes seen this episode
    
    episode_score = 0
    
    for t in xrange(200):
        env.render()
        #print observation
        
        old_state = observation  # retain old state for updates
        old_state = np.reshape(old_state,(4, 1))  # reshape into numpy array
        
        action = env.action_space.sample()  # initialize random action
        
        if wondering_gnome.have_archetypes:
        
            old_archetype = wondering_gnome.get_state_archetype(old_state)  # get old state's arch
        
            random_fate = np.random.random()
            
            if random_fate > wondering_gnome.epsilon:  # e-greedy implementation 
            
                if wondering_gnome.is_arch_in_sas_memory(old_archetype):
                    
                    action = wondering_gnome.get_best_action(old_archetype)
                        
        
        observation, reward, done, info = env.step(action)
        
        new_state = observation  
        new_state = np.reshape(new_state,(4, 1))  # reshape into numpy array
        
        episode_score += reward
        
        if wondering_gnome.have_archetypes:
        
            episode_archetypes.append(old_archetype)
        
            new_archetype = wondering_gnome.get_state_archetype(new_state)  # get new state's arch
        
            sas_tuple = (old_archetype, action, new_archetype)

            wondering_gnome.add_to_sas_mem_if_needed(sas_tuple)
        
        
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
    
    if wondering_gnome.have_archetypes:
    
        if i_episode % 5 == 0:  # rebalance archetypes every so often
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(wondering_gnome.archetypes, wondering_gnome.memory) 
            
            print "#3"
            assert not len(wondering_gnome.archetypes) < wondering_gnome.number_of_archetypes
        
        #update archetype values
        wondering_gnome.update_archetype_values(episode_archetypes, episode_score)
        
            
    if wondering_gnome.should_we_exploit():
    
        if not wondering_gnome.have_archetypes:
            
            #  initialize archetypes
            some_archetypes = wondering_gnome.get_archetypes(wondering_gnome.number_of_archetypes, wondering_gnome.memory)
            
            print "#1"
            assert not len(some_archetypes) < wondering_gnome.number_of_archetypes
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(some_archetypes, wondering_gnome.memory)
    
            print "#2"
            assert not len(wondering_gnome.archetypes) < wondering_gnome.number_of_archetypes    
    
            wondering_gnome.have_archetypes = True
            
            for index in range(wondering_gnome.number_of_archetypes):
            
                wondering_gnome.arch_value_memory[index] = 0  # initialize archetype values
    
        
        wondering_gnome.decay_epsilon()  
             
             
print rewards_list             
    
print wondering_gnome.archetypes 

print wondering_gnome.arch_value_memory

print wondering_gnome.transition_memory

print np.average(rewards_list)
    
    
    
    
    
    # model-building state value iteration agent, using archetypes

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
        self.number_of_archetypes = 6  # number of archetypes to use for our states
        self.learning_rate = 0.7  # how quickly do we learn
        self.memory_b4_exploit = 200  # how much memory before exploiting 
        self.max_memory = 3e4  # maximum length of states to store in memory
        self.iteration = 0  # how many states have we seen 
        self.have_archetypes = False  # bool, do we have archetypes yet
        self.foresight = 3  # how many actions do we plan ahead
        
        #memories
        self.transition_memory = []  # list for our SAS transitions 
        self.arch_value_memory = {}  # dict of pairs indicating the value of our archetypes
        self.archetypes = {}  # dict of our archetypes, key is their index, value is their array representation 
        self.memory = []  # memory of states from which we form our archetypes    
    
    
    
    def add_to_sas_mem_if_needed(self, sas_tuple):
    
        if not sas_tuple[0] == sas_tuple[2]:  # is this a transition 
    
            if not sas_tuple in self.transition_memory:  # do we already know it
                
                added_bool = False
                
                for index, sas in enumerate(self.transition_memory):
                    
                    current_len = len(self.transition_memory)
                    
                    if (sas[0],sas[1]) == (sas_tuple[0],sas_tuple[1]):
                        
                        old_sas = self.transition_memory.pop(index)  # prune old transistion memories

                        self.transition_memory.append(sas_tuple)
                        
                        assert current_len == len(self.transition_memory)
                        
                        added_bool = True
            
                if not added_bool:

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
        
        assert not len(possible_transitions) == 0
        
        best_value = -1 # initialize
        
        best_transition = None # initialize        
                                
        for transition in possible_transitions:
        
            value_of_transition = self.arch_value_memory[transition[2]]  # value of next archetype given action
            
            if value_of_transition > best_value:
            
                best_value = value_of_transition
                
                best_transition = transition
                
        assert not best_transition == None
                
        best_action = best_transition[1]  # get action from best transition 
        
        return best_action
        
                 
    
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
         
            raw_arch = self.archetypes[archetype]  #  get array of archetype
            
            dist = self.get_L2_distance(raw_arch, current_state)  
            
            if dist < closest_distance:
            
                closest_distance = dist
                
                closest_arch = archetype
                
        assert not closest_arch == None
        
        return closest_arch  # returning archetype key
                
    
    
    def get_L2_distance(self, state1, state2):
    
        return np.linalg.norm(state1-state2)
        
        
        
    #def
    #def
    


# gym env
env = gym.make('CartPole-v0')

wondering_gnome = value_iter_agent(env.action_space)

rewards_list = []

for i_episode in xrange(400):
    observation = env.reset()
    
    episode_archetypes = []  # archetypes seen this episode
    
    episode_score = 0
    
    for t in xrange(200):
        env.render()
        #print observation
        
        old_state = observation  # retain old state for updates
        old_state = np.reshape(old_state,(4, 1))  # reshape into numpy array
        
        action = env.action_space.sample()  # initialize random action
        
        if wondering_gnome.have_archetypes:
        
            old_archetype = wondering_gnome.get_state_archetype(old_state)  # get old state's arch
        
            random_fate = np.random.random()
            
            if random_fate > wondering_gnome.epsilon:  # e-greedy implementation 
            
                if wondering_gnome.is_arch_in_sas_memory(old_archetype):
                    
                    action = wondering_gnome.get_best_action(old_archetype)
                        
        
        observation, reward, done, info = env.step(action)
        
        new_state = observation  
        new_state = np.reshape(new_state,(4, 1))  # reshape into numpy array
        
        episode_score += reward
        
        if wondering_gnome.have_archetypes:
        
            episode_archetypes.append(old_archetype)
        
            new_archetype = wondering_gnome.get_state_archetype(new_state)  # get new state's arch
        
            sas_tuple = (old_archetype, action, new_archetype)

            wondering_gnome.add_to_sas_mem_if_needed(sas_tuple)
        
        
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
    
    if wondering_gnome.have_archetypes:
    
        if i_episode % 5 == 0:  # rebalance archetypes every so often
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(wondering_gnome.archetypes, wondering_gnome.memory) 
            
            print "#3"
            assert not len(wondering_gnome.archetypes) < wondering_gnome.number_of_archetypes
        
        #update archetype values
        wondering_gnome.update_archetype_values(episode_archetypes, episode_score)
        
            
    if wondering_gnome.should_we_exploit():
    
        if not wondering_gnome.have_archetypes:
            
            #  initialize archetypes
            some_archetypes = wondering_gnome.get_archetypes(wondering_gnome.number_of_archetypes, wondering_gnome.memory)
            
            print "#1"
            assert not len(some_archetypes) < wondering_gnome.number_of_archetypes
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(some_archetypes, wondering_gnome.memory)
    
            print "#2"
            assert not len(wondering_gnome.archetypes) < wondering_gnome.number_of_archetypes    
    
            wondering_gnome.have_archetypes = True
            
            for index in range(wondering_gnome.number_of_archetypes):
            
                wondering_gnome.arch_value_memory[index] = 0  # initialize archetype values
    
        
        wondering_gnome.decay_epsilon()  
             
             
print rewards_list             
    
print wondering_gnome.archetypes 

print wondering_gnome.arch_value_memory

print wondering_gnome.transition_memory

print np.average(rewards_list)
    
    
    
    
    
    
    
    
            
       
    
    
            
       