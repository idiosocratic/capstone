# model-free discrete policy gradient agent using archetypes


# if we_did_well:
#   
#     look at the actions we took for each archetype
#     average actions over all choices
#     update archetype preferred action based on learning rate and averaged action
    
    
import numpy as np
import gym
#import archetypes #functions: get_archetypes & rebalance_archetypes
import random


class poly_grad_agent(object):

    def __init__(self, action_space):
    
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Hey, not our space!'
        
        
        #hyperparameters, some aren't that hyper
        self.epsilon = 0.37  # how much do we explore
        self.epsilon_decay = 0.95  # rate at which we decay epsilon
        self.number_of_archetypes = 12  # number of archetypes to use for our states
        self.learning_rate = 0.01  # how quickly do we learn
        self.memory_b4_exploit = 200  # how much memory before exploiting 
        self.max_memory = 3e4  # maximum length of states to store in memory
        self.iteration = 0  # how many states have we seen 
        self.have_archetypes = False  # bool, do we have archetypes yet
        self.average_score = 0  # keep track of score to know if we're doing well
        self.base = 0  # base value for deciding actions
        
        #memories
        self.arch_action_memory = {}  # dict of pairs indicating the best action of our archetypes
        self.archetypes = {}  # dict of our archetypes, key is their index, value is their array representation 
        self.memory = []  # memory of states from which we form our archetypes    
        self.episode_scores  # keep scores in memory for calculating average
    
    
    
    def did_we_do_well(self, episode_score):
    
        if episode_score > self.average_score:  # better than average or not
    
            return True
        
        return False  
    
    
    
    def update_archetype_actions(self, episode_archetype_actions):
    
        for tuple in episode_archetype_actions:  # expecting (archetype, action)  
            
            action_weight = None  # initialize action
            
            archetype = tuple[0]
            
            if tuple[1] == 1:
            
                action_weight = 1       
                
            if tuple[1] == 0:
            
                action_weight = -1     
                
            assert not action_weight == None    
            
            updated = False
            
            for arch_key in self.arch_action_memory:
            
                if archetype == arch_key:  
                
                    self.arch_action_memory[arch_key] += (self.learning_rate * action_weight)
                    
                    updated = True
             
            assert updated
            
                
       
    def get_archetype_action(self, current_archetype):  # get best action given the archetype of our current states
        
        found_arch = False
        
        for arch in self.arch_action_memory:
            
            if arch == current_archetype:
            
                #found_arch = True
                
                return self.arch_action_memory[arch]
                
        assert found_arch        
        
          
                 
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

        for index, child_list in enumerate(archetype_child_lists):
            
            if not len(child_list) == 0:
            
                new_archetype = sum(child_list)/float(len(child_list)) # get average of child_list            
                
                new_archetypes.append(new_archetype)
                
            if len(child_list) == 0:    
            
                new_archetype = old_archetypes[index]  # this one has no children for updating
                
                new_archetypes.append(new_archetype)
            
        
        new_archetypes_dict = {}
        
        for index, arch in enumerate(new_archetype):  # populate our dict 
        
            new_archetypes_dict[index] = arch
        
        return new_archetypes_dict
        
        
                
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

wondering_gnome = poly_grad_agent(env.action_space)


for i_episode in xrange(200):
    observation = env.reset()
    
    episode_archetypes = []  # archetypes seen this episodes
    
    episode_score = 0
    
    for t in xrange(200):
        env.render()
        print observation
        
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
        
        
        
        wondering_gnome.memory.append(old_state)
        
        print "Old state, action, new state, reward: "
        print old_state, action, new_state, reward
        print "Shape: "
        print observation.shape
        
        
        wondering_gnome.iteration += 1
        
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break
     
    
    if wondering_gnome.have_archetypes:
    
        if i_episode % 5 == 0:  # rebalance archetypes every so often
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(wondering_gnome.archetypes, wondering_gnome.memory) 
            
        
        #update archetype values
        wondering_gnome.update_archetype_values(episode_archetypes, episode_score)
        
            
    if wondering_gnome.should_we_exploit():
    
        if not wondering_gnome.have_archetypes:
            
            #  initialize archetypes
            some_archetypes = wondering_gnome.get_archetypes(wondering_gnome.number_of_archetypes, wondering_gnome.memory)
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(some_archetypes, wondering_gnome.memory)
    
            wondering_gnome.have_archetypes = True
            
            for index in range(wondering_gnome.number_of_archetypes):
            
                wondering_gnome.arch_value_memory[index] = 0  # initialize archetype values
    
        
        wondering_gnome.decay_epsilon()  
             
    
    
    
    
    
    
    
    
    
    
            
           