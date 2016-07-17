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
        self.epsilon = 0.63  # how much do we explore
        self.epsilon_decay = 0.99  # rate at which we decay epsilon
        self.number_of_archetypes = 6  # number of archetypes to use for our states
        self.learning_rate = 0.3  # how quickly do we learn
        self.memory_b4_exploit = 100  # how much memory before exploiting 
        self.max_memory = 3e4  # maximum length of states to store in memory
        self.iteration = 0  # how many states have we seen 
        self.have_archetypes = False  # bool, do we have archetypes yet
        self.average_score = 0  # keep track of score to know if we're doing well
        self.high_score = 0  # keep track of high score to know if we're doing well
        self.lowest_score = 0  # keep track of lowest score to know if we're doing well
        self.base = 0  # base value for deciding actions
        
        #memories
        self.arch_action_memory = {}  # dict of pairs indicating the best action of our archetypes
        self.archetypes = {}  # dict of our archetypes, key is their index, value is their array representation 
        self.memory = []  # memory of states from which we form our archetypes    
        self.episode_scores = [] # keep scores in memory for calculating average
    
    
    
    def did_we_do_well(self, episode_score):
    
        if episode_score > self.high_score * 0.7:  # close to high score
    
            return True
        
        return False  
    
    
    def did_we_do_poorly(self, episode_score):
    
        if episode_score < self.lowest_score * 2:  # close to lowest score
    
            return True
        
        return False  
    
    
    def update_average_score(self):
    
        self.average_score = 1.33 * np.average(self.episode_scores)
        
        
        
    def update_high_score(self):
    
        self.high_score = np.max(self.episode_scores)  
        
        
        
    def update_lowest_score(self):
    
        self.lowest_score = np.min(self.episode_scores)            
    
    
    
    def correct_our_actions(self, archetype_action_list):
        
        new_arch_action_list = []
    
        for arch_action in archetype_action_list:
            
            corrected_action = None
            
            if arch_action[1] == 1:
            
                corrected_action = 0
                
            if arch_action[1] == 0:
            
                corrected_action = 1
            
            assert not corrected_action == None    
                
            new_arch_action = (arch_action[0], corrected_action)    
            
            new_arch_action_list.append(new_arch_action)
                
        return new_arch_action_list            
    
    
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
                
                if self.arch_action_memory[arch] > self.base:
                
                    return 1
                    
                if self.arch_action_memory[arch] < self.base:
                
                    return 0    
                    
                if self.arch_action_memory[arch] == self.base:    
                
                    return random.choice([0,1])
                
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
        
        for index, arch in enumerate(new_archetypes):  # populate our dict 
        
            new_archetypes_dict[index] = arch
            
            
        assert len(old_archetypes) == len(new_archetypes_dict)    
        
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


for i_episode in xrange(250):
    observation = env.reset()
    
    #episode_archetypes = []  # archetypes seen this episodes
    
    episode_archetype_actions = []  # tuples (archetype, action)
    
    episode_score = 0
    
    exploring_dict  = {}
    
    if wondering_gnome.have_archetypes:
        
        for archetype_key in wondering_gnome.archetypes:

            exploring_dict[archetype_key] = env.action_space.sample()
    
    exploring = False
    
    random_fate = np.random.random()
            
    if random_fate < wondering_gnome.epsilon:
        
        exploring = True
    
    
    for t in xrange(200):
        env.render()
        #print observation
        
        old_state = observation  # retain old state for updates
        old_state = np.reshape(old_state,(4, 1))  # reshape into numpy array
        
        action = env.action_space.sample()  # initialize random action
        
        if wondering_gnome.have_archetypes:
        
            old_archetype = wondering_gnome.get_state_archetype(old_state)  # get old state's arch
            
            if exploring:  # e-greedy implementation 
            
                action = exploring_dict[old_archetype]
                        
        
        observation, reward, done, info = env.step(action)
        
        new_state = observation  
        #new_state = np.reshape(new_state,(4, 1))  # reshape into numpy array
        
        episode_score += reward
        
        if wondering_gnome.have_archetypes:
        
            #episode_archetypes.append(old_archetype)
        
            episode_archetype_actions.append((old_archetype,action))
        
        
        if not len(wondering_gnome.memory) > wondering_gnome.max_memory: # have we reached our maximum memory
        
            wondering_gnome.memory.append(old_state)
        
        #print "Old state, action, new state, reward: "
        #print old_state, action, new_state, reward
        #print "Shape: "
        #print observation.shape
        
        
        wondering_gnome.iteration += 1
        
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break
     
    wondering_gnome.episode_scores.append(episode_score)
    
    wondering_gnome.update_high_score()
    
    wondering_gnome.update_average_score()
    
    wondering_gnome.update_lowest_score()
    
    if wondering_gnome.have_archetypes:
    
        if i_episode % 5 == 0:  # rebalance archetypes every so often
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(wondering_gnome.archetypes, wondering_gnome.memory) 
        
            
    if wondering_gnome.should_we_exploit():
    
        if not wondering_gnome.have_archetypes:
            
            #  initialize archetypes
            some_archetypes = wondering_gnome.get_archetypes(wondering_gnome.number_of_archetypes, wondering_gnome.memory)
            
            wondering_gnome.archetypes = wondering_gnome.rebalance_archetypes(some_archetypes, wondering_gnome.memory)
    
            wondering_gnome.have_archetypes = True
            
            for index in range(wondering_gnome.number_of_archetypes):
            
                wondering_gnome.arch_action_memory[index] = wondering_gnome.base  # initialize archetype actions
    
        if not len(episode_archetype_actions) == 0:
            
            Added = False
            
            if wondering_gnome.did_we_do_well(episode_score):
        
                wondering_gnome.update_archetype_actions(episode_archetype_actions)
                
                Added = True
                
            if not Added:
            
                if wondering_gnome.did_we_do_poorly(episode_score):
                    print "Poorly: "
                    print episode_archetype_actions

                    episode_archetype_actions = wondering_gnome.correct_our_actions(episode_archetype_actions)
                
                    print episode_archetype_actions
                    print "Poorly ^^^ "
                    wondering_gnome.update_archetype_actions(episode_archetype_actions)
                
                    #assert False 
             
        
        wondering_gnome.decay_epsilon()  
        
        print "Episode #: "
        print i_episode
        print "Epsilon: "
        print wondering_gnome.epsilon

        print "actions"
        print wondering_gnome.arch_action_memory

        
print wondering_gnome.average_score   
print "last 30:"
print np.average(wondering_gnome.episode_scores[-30:])
print "30"    
     
dist_list = []     
     
for arch1 in wondering_gnome.archetypes:
    
    dist_sum = 0 
     
    for arch2 in wondering_gnome.archetypes:
        
        dist = 0 
        
        if not arch1 == arch2:
        
            archy1 = wondering_gnome.archetypes[arch1]
            
            archy2 = wondering_gnome.archetypes[arch2]
        
            dist = wondering_gnome.get_L2_distance(archy1,archy2)
            
            dist_sum += dist

    dist_list.append(dist_sum)    
    
    
print "avg dist"
print list(np.array(dist_list)/float(wondering_gnome.number_of_archetypes))    

print "actions"
print wondering_gnome.arch_action_memory
        
    
    
    
    
    
        
    
    
    
    
    
    
            
           