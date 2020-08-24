import argparse
import sys

import gym
from gym import wrappers, logger
import numpy as np
import tensorflow as tf
import skimage.measure


class LogisticAgent(object):
    def __init__(self, env, action_space, observation_sample, learning_rate, epsilon, discount):
        self.env = env
        
        self.action_space = action_space
        self.action_space_dim = action_space.n
        
        self.logit_count = observation_sample.flatten().shape[0]
        
        self.weights = 1 / np.sqrt(self.logit_count) * np.random.randn(self.action_space_dim, self.logit_count)
        
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount = discount
    

    def train(self, episode_count):
        for i in range(episode_count):
            episode_reward = 0
            state = env.reset().flatten()
            state_value, action = self.get_action(state, do_exploration = True)
            
            while True:
                new_state, reward, done, _ = self.env.step(action)
                new_state = new_state.flatten()
                episode_reward += reward
                
                if done:
                    self.update_weights(reward, np.zeros((self.action_space_dim, 1)), state_value, state, action)
                    break
                
                new_state_value, new_action = self.get_action(new_state, do_exploration = True)
                self.update_weights(reward, new_state_value, state_value, state, action)
                
                state = new_state
                action = new_action
            
            print("Episode reward is " + str(episode_reward))
            
            
    
    def get_action(self, observation, do_exploration = False):
        state_value = np.matmul(self.weights, observation)
        
        if do_exploration and np.random.uniform() < self.epsilon:
            return state_value, self.action_space.sample()
        else:
            return state_value, np.argmax(state_value)
    

    def update_weights(self, reward, new_state_value, old_state_value, old_state, old_action):
        td_target = reward + self.discount * np.max(new_state_value)
        td_difference = td_target - old_state_value[old_action]
        
        self.weights[old_action, :] += self.learning_rate * td_difference * old_state
        summed_weights = np.sum(self.weights, axis = 0)
        self.weights /= summed_weights
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MsPacman-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    #outdir = '/tmp/random-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    starting_ob = env.reset()
    
    agent = LogisticAgent(env, env.action_space, starting_ob, 0.5, 0.5, 0.9)

    episode_count = 100
    
    agent.train(episode_count)

    # Close the env and write monitor result info to disk
    env.close()