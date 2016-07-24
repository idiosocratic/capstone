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
        self.epsilon = 0.53  # how much do we explore
        self.epsilon_decay = 0.9995  # rate at which we decay epsilon
        self.number_of_archetypes = 8  # number of archetypes to use for our states
        self.memory_b4_arch_initialize = 200  # how much memory before exploiting
        self.max_memory = 3e4  # maximum length of states to store in memory
        self.iteration = 0  # how many states have we seen 
        self.have_archetypes = False  # bool, do we have archetypes yet
        
        #memories
        self.highest_score = 0  # highest score received so far 
        self.highest_permutation = {}  # dictionary of the best archetype_actions discovered so far
        self.arch_action_memory = []  # list of archetype action dicts
        self.archetypes = {}  # dict of our archetypes, key is their index, value is their array representation 
        self.memory = []  # memory of states from which we form our archetypes    
        self.action_superset = []  # superset of archetype actions


    def should_we_initialize_archs(self):
    
        if self.iteration > self.memory_b4_arch_initialize:
    
            return True

        return False
    


    def create_arch_action_superset(self, possible_actions):

        initial_arch_action_list = []
        
        print "actions"
        print possible_actions[0]
        print "actions"
        print possible_actions[1]
        print "actions"
        
        zero = possible_actions[0]
        print zero
        
        
        for i in range(self.number_of_archetypes):

            initial_arch_action_list.append(zero)
        
        
        print initial_arch_action_list
        
        self.permute_next_index(initial_arch_action_list, 0, possible_actions)
        
        #dedup_action_
        
        #self.action_superset = list(set(self.action_superset))
        
        print possible_actions
        print initial_arch_action_list
        print "Len: "
        print len(self.action_superset)
        
        assert len(self.action_superset) >= len(possible_actions)**(self.number_of_archetypes)
        
        arch_action_dict_superset = []
            
        for action_list in self.action_superset:
            
            dict_index = len(arch_action_dict_superset)
            
            arch_action_dict_superset.append({})
            
            for action_index, action in enumerate(action_list):

                arch_action_dict_superset[dict_index][action_index] = action
                
        
        self.arch_action_memory = arch_action_dict_superset
                
                
                
    def permute_next_index(self, action_list, index_to_permute, action_set):
    
        if index_to_permute < len(action_list):
            
            permutation_children = []
            
            for num in range(len(action_set)):
            
                permutation_children.append([])  # initialize permutations
        

            for action in action_list:
            
                for entry in permutation_children:
                
                    entry.append(action)
            
            
            for index, action in enumerate(action_set):
                
                permutation_children[index][index_to_permute] = action

            
            for child in permutation_children:

                if not child in self.action_superset:

                    self.action_superset.append(child)
        
                    print "added"


            next_permutation_index = index_to_permute + 1


            print "Permute children: "
            print permutation_children
            
            for child in permutation_children:
                
                self.permute_next_index(child, next_permutation_index, action_set)



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
    
        if episode_score >= self.highest_score:
            
            print "Replaced: " + str(self.highest_permutation)
            print "For: " + str(arch_action_dict)
            print "@ " + str(episode_score)
            
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

wondering_gnome.create_arch_action_superset([0,1])

rewards_list = []

for i_episode in xrange(400):
    observation = env.reset()
    
    #episode_archetypes = []  # archetypes seen this episode
    
    episode_score = 0
    
    epi_arch_action_dict = wondering_gnome.highest_permutation  # initialize our action dict
    
    if wondering_gnome.should_we_explore():
    
        epi_arch_action_dict = random.choice(wondering_gnome.arch_action_memory)  # we're exploring
        print "Exploring!"
    
    print epi_arch_action_dict
    
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
        
            
        wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(some_archetypes, wondering_gnome.memory)
    
        print "#2"
        assert not len(wondering_gnome.archetypes) < wondering_gnome.number_of_archetypes
    
        wondering_gnome.have_archetypes = True
        

             
             
print rewards_list             
    
print wondering_gnome.archetypes 

print wondering_gnome.highest_permutation

print wondering_gnome.highest_score

print np.average(rewards_list)

print wondering_gnome.action_superset
    
    
