# nearest neighbor majority vote implementation


import numpy as np
import gym


class NnSelcMemAgent(object):

    def __init__(self, action_space):
    
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Yo, not our space!'
        
        # hyperparameters
        self.epsilon = 0.9  # exploration percentage
        self.epsilon_decay = 0.99 # exploration decay
        self.similar_state_mem_num = 13 # number of similar states to keep in memory
        self.similarity_threshold = 0.1 # how close states can be before we consider them roughly equivalent
        self.state_action_rewards_memory = [] # selective memory for our (state,action,reward) tuples
        self.iteration = 0 # how many actions have we taken
        self.time_before_exploit = 337 # how much knowledge to build before using it
    
    
    
    def add_to_mem_if_needed(self, episode_state_action_reward_tuple_list):
        
        
        for s_a_r_1 in episode_state_action_reward_tuple_list:
        
            episode_state = s_a_r_1[0]
            
            episode_action = s_a_r_1[1]
            
            episode_reward = s_a_r_1[2]
            
            state_added = False
            
            number_of_sim_states_in_mem = 0
            
            index_of_sim_state_with_lowest_reward = None
            
            lowest_reward_for_sim_mem_state = 1e4  # initialize as very high value
            
            for index, s_a_r_2 in enumerate(self.state_action_rewards_memory):
                
                mem_state = s_a_r_2[0]
                
                mem_rewards = s_a_r_2[2]
                
                mem_state_index = index
                
                if self.are_these_states_the_same(state1,state2):
                
                    number_of_sim_states_in_mem +=1    
                        
                    if mem_rewards < lowest_reward_for_sim_mem_state:
        
                        index_of_sim_state_with_lowest_reward = mem_state_index
        
        
            if number_of_sim_states_in_mem < self.similar_state_mem_num : # do we need more memory for this state  
            
                self.state_action_rewards_memory.append(s_a_r_1)
                
                state_added = True
                
            if not state_added: # have enough memory, but is it good enough
            
                if lowest_reward_for_sim_mem_state < episode_rewards:
    
                    pruned_mem = self.state_action_rewards_memory.pop(index_of_sim_state_with_lowest_reward)
                    
                    self.state_action_rewards_memory.append(s_a_r_1)
                


    def are_these_states_the_same(self, state1, state2):
        
        distance = self.get_L2_distance(state1, state2)
            
        if distance < self.similarity_threshold:
        
            return True
            
        return False
        
        
                    
    def get_best_action(self, state):
    
        nearest_sar_tuples = self.get_closest_states(state)
        
        action_list = [] # get the actions from similar states
        
        for s_a_r in nearest_sar_tuples:
        
            action_list.append(s_a_r[1])
                    
        averaged_action = np.average(action_list) # get majority vote           
                    
        action = None # initialize
            
        if averaged_action >= 0.5: # round
            
            action = 1 
                
        if averaged_action < 0.5: # round
            
            action = 0    
                
        assert not (action == None) # assert have action
        
        return action
            
    
    
    def get_L2_distance(self, state1, state2):
        
        return np.linalg.norm(state1-state2)
        
        
    
    def get_closest_states(self, state):  
        
        nearest_sar_tuples = []
        
        for s_a_r in self.state_action_rewards_memory:
        
            if self.are_these_states_the_same(s_a_r[0], state):
            
                nearest_sar_tuples.append(s_a_r)
        
        return nearest_sar_tuples       
              
        
            
    def should_we_exploit(self):    
    
        if self.iteration > self.time_before_exploit: 
            
            return True  # we have enough knowledge
    
        return False  # not enough knowledge to exploit 
    
    
        
    def decay_epsilon(self):
    
        self.epsilon *= self.epsilon_decay

    
  


env = gym.make('CartPole-v0')
wondering_gnome = NnSelcMemAgent(env.action_space)        
            
episode_rewards_list = []            
            
for i_episode in xrange(200):
    observation = env.reset()
    
    episode_rewards = 0
    #episode_state_list = []
    episode_state_action_list = []
    
    for t in xrange(200):
        env.render()
        
        current_state = observation  
        
        action = env.action_space.sample() # initialize action randomly
        
        # should we override action 
        if wondering_gnome.should_we_exploit():
            
            random_fate = np.random.random()
             
            if random_fate > wondering_gnome.epsilon: # epsilon greedy implementation
           
                action = wondering_gnome.get_best_action(current_state) # overwrite random action 

                print "Exploiting!!!!!!!!!!!!!!"
                #if action == None:
                
                #    action = env.action_space.sample()
                    
        
        observation, reward, done, info = env.step(action)
        
        new_state = observation
        
        episode_rewards += reward
        
        wondering_gnome.iteration += 1
        
        #episode_state_list.append(current_state)
        
        episode_state_action_list.append((current_state, action))
        
        print "Iteration_number: "
        print wondering_gnome.iteration 
        print "Epsilon: "
        print wondering_gnome.epsilon
 
        if done:
            
            print "Episode finished after {} timesteps".format(t+1)
            break
                
                
    print "Episode Rewards: "
    print episode_rewards
    episode_rewards_list.append(episode_rewards)
    
    episode_state_action_reward_tuples = []
    
    for state_action in episode_state_action_list:
    
        episode_state_action_reward_tuples.append((state_action[0],state_action[1],episode_rewards))
        
    wondering_gnome.add_to_mem_if_needed(episode_state_action_reward_tuples)
    
    if wondering_gnome.should_we_exploit():

        wondering_gnome.decay_epsilon()
    
           
      
print "Rewards List: "
print episode_rewards_list
print "Average Overall: "
print np.average(episode_rewards_list)   
print "Average of last 30 episodes: "
print np.average(episode_rewards_list[-30:]) 
print len(wondering_gnome.state_action_rewards_memory)

