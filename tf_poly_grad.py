
#### Libraries
# Standard library
import random
from collections import deque

# Third-party libraries
import numpy as np
import tensorflow as tf
import gym


class PolyGradAgent(object):
    
    def __init__(self, action_space):
        
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Yo, not our space!'

        ### model hyperparameters

        self.epsilon = 0.7  # how much do we explore
        self.epsilon_decay_rate = 0.99  # rate by which exploration decreases
        self.high_score = 0  # keep track of highest score obtained thus far
        self.did_well_threshold = 0.77  # how close we need to be to our high score to have "done well"
        self.net_trained = False  # has our neural net had any training

        ### finite state_action memory for episodes where we did well
        self.state_action_mem = deque(maxlen = 5000)



    def get_next_batch(self, batch_size):
            
        if len(self.state_action_mem) < batch_size:
        
            assert False, "Not enough memory for batch size!"
            
        rand_batch = random.sample(self.state_action_mem, batch_size)  # get random sample
    
        _states = []
        _actions = []
    
        for state_action in rand_batch:  # format random sample
        
            _states.append(state_action[0])
            _actions.append(state_action[1])
    
        batch_lists = (_states, _actions)

        return batch_lists



    def did_we_do_well(self, episode_rewards):

        if episode_rewards > self.did_well_threshold * self.high_score:

            return True

        return False



    def add_to_memory(self, episode_state_action_list):

        for state_action in episode_state_action_list:

            self.state_action_mem.append(state_action)



    def are_we_exploring(self):

        if not self.net_trained:

            return True

        if self.net_trained:

            random_fate = np.random.random()

            if random_fate > self.epsilon:  # e-greedy implementation

                return True
        
        return False



    def update_high_score(self, episode_rewards):

        if episode_rewards > self.high_score:

            self.high_score = episode_rewards



    def decay_epsilon(self):

        self.epsilon *= self.epsilon_decay_rate



### set up tensorflow neural net

def weight_variable(shape):
    
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


sess = tf.InteractiveSession()  # initialize tensorflow session


state = tf.placeholder(tf.float32,[None,4])
actions = tf.placeholder(tf.float32,[None,2])
w1 = weight_variable([4,10])
b1 = bias_variable([10])
h1 = tf.nn.relu(tf.matmul(state,w1) + b1)

keep_prob = tf.placeholder(tf.float32)
h1_drop = tf.nn.dropout(h1, keep_prob)

w2 = weight_variable([10,2])
b2 = bias_variable([2])
output = tf.nn.softmax(tf.matmul(h1_drop,w2) + b2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(actions * tf.log(output), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(actions,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())


### Save the variables to disk. (In case we want to store our model)
#save_path = saver.save(sess, "/Users/idiosocratic/desktop/env/capstone/model.ckpt")
#print("Model saved in file: %s" % save_path)


env = gym.make('CartPole-v0')
wondering_gnome = PolyGradAgent(env.action_space)

for i_episode in xrange(2000):
    observation = env.reset()
    episode_rewards = 0
    episode_state_action_list = []
    
    #episode_state_target_list = []
    for t in xrange(200):
        env.render()
        #print observation
        
        current_state =  np.expand_dims(observation, axis=0)
        
        raw_output = output.eval(feed_dict = {state: current_state, keep_prob: 1.0})
        
        action = np.argmax(raw_output)
        
        #action_list.append(action)
        
        observation, reward, done, info = env.step(action)
        
        episode_state_action_list.append((current_state, action))
        
        episode_rewards += reward
        
        #iteration_number += 1
        #print "iteration_number: "
        #print iteration_number
        
        if done:
            
            #print "Episode finished after {} timesteps".format(t+1)
            break

    print "E Rs: "
    print episode_rewards

    if wondering_gnome.did_we_do_well(episode_rewards):

        wondering_gnome.add_to_memory(episode_state_action_list)


    batch_size = 100

    if len(wondering_gnome.state_action_mem) > batch_size:

        batch = wondering_gnome.get_next_batch(batch_size)

        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={ state:batch[0], actions: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
                
        train_step.run(feed_dict={state: batch[0], actions: batch[1], keep_prob: 0.75})





