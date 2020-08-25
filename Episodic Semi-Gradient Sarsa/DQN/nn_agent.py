import argparse
import sys

import torch
import torch.optim as optim
import torch.nn.functional as F

import gym
from gym import wrappers, logger
import numpy as np
import random

from model import DQN
from replay_memory import ReplayMemory, Transition

class DQNAgent(object):
    def __init__(self, env, action_space, observation_sample, epsilon, discount, batch_size, replay_memory_size):
        self.env = env
        self.device = torch.device("cpu")
        
        self.action_space = action_space
        self.action_space_dim = action_space.n
        
        self.input_shape = observation_sample.shape
        self.h = self.input_shape[0]
        self.w = self.input_shape[1]
        
        self.policy_net = DQN(self.h, self.w, self.action_space_dim).to(self.device)
        self.target_net = DQN(self.h, self.w, self.action_space_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(self.replay_memory_size)
        
        self.EPSILON = epsilon
        self.DISCOUNT = discount
        self.BATCH_SIZE = batch_size
    
    def transpose_to_torch(self, state):
        state = np.transpose(state, (2, 0, 1))
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        state = torch.from_numpy(state)
        return state.unsqueeze(0).to(self.device)
    
    def fill_replay_memory(self):
        transitions_gathered = 0
        while True:
            
            state = env.reset()
            state = self.transpose_to_torch(state)
            episode_reward = 0
            
            while True:
                action = self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.transpose_to_torch(next_state)
                
                episode_reward += reward
                reward = torch.tensor([reward], device=self.device)
                
                if done:
                    self.memory.push(state, action, None, reward)
                    transitions_gathered += 1
                    print("Episode reward is: ", episode_reward)
                    break
                else:
                    self.memory.push(state, action, next_state, reward)

                state = next_state
                transitions_gathered += 1
            
            if transitions_gathered >= self.replay_memory_size:
                break
    
    def train(self, episode_count):
        # Run several episodes before replay memory is full
        self.fill_replay_memory()

        for i in range(episode_count):
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
            state = env.reset()
            state = self.transpose_to_torch(state)
            episode_reward = 0
            
            while True:
                action = self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.transpose_to_torch(next_state)
                
                episode_reward += reward
                reward = torch.tensor([reward], device=self.device)
                
                if done:
                    self.memory.push(state, action, None, reward)
                    print("Episode reward is: ", episode_reward)
                    break
                else:
                    self.memory.push(state, action, next_state, reward)
                    
                self.optimize_model()

                state = next_state
    
    
    def get_action(self, state, do_exploration = True):
        if do_exploration and np.random.uniform() < self.EPSILON:
            return torch.tensor([[random.randrange(self.action_space_dim)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        
    def optimize_model(self, batch_count = 1):
        for i in range(batch_count):
            #print(i)
            transitions = self.memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
        
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.DISCOUNT) + reward_batch
        
            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        

BATCH_SIZE = 32
DISCOUNT = 0.9
EPSILON = 0.1
REPLAY_MEMORY_SIZE = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MsPacman-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id).unwrapped

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    #outdir = '/tmp/random-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    starting_ob = env.reset()
    
    agent = DQNAgent(env, env.action_space, starting_ob, EPSILON, DISCOUNT, BATCH_SIZE, REPLAY_MEMORY_SIZE)

    episode_count = 100
    
    agent.train(episode_count)

    # Close the env and write monitor result info to disk
    env.close()