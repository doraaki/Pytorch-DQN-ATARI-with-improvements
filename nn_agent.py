import argparse

import torch

import gym
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import json

from model import DQN
from replay_memory import ReplayMemory
from policy import Policy

class DQNAgent(object):
    def __init__(self, env, config):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_actions = env.action_space.n
        self.scaled_image_height = config['atari']['scaled_image_height']
        self.scaled_image_width = config['atari']['scaled_image_width']
        
        self.policy_net = DQN(config, self.num_actions).to(self.device)
        self.target_net = DQN(config, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.replay_memory_size = config['rl_params']['replay_memory_size']
        self.replay_memory = ReplayMemory(self.replay_memory_size)
        
        self.policy = Policy(config, self.num_actions, self.policy_net, self.target_net, self.device, self.replay_memory)
        
        self.evaluation_episodes_count = config['rl_params']['evaluation_episodes_count']
        self.steps_between_evaluations = config['rl_params']['steps_between_evaluations']
        
        self.frames_stacked = config['atari']['frames_stacked']
        
        self.weights_path = config['train']['weights.pth']
    
    def transpose_to_torch(self, state):
        state = np.transpose(state, (2, 0, 1))
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        state = torch.from_numpy(state)
        return state.unsqueeze(0).to(self.device)
    
    def rescale_screen(self, screen):
        screen = cv2.resize(screen, dsize=(self.scaled_image_height,self.scaled_image_width))
        return screen
    
    def reset_env(self):
        self.policy.set_parameters_for_new_episode()
        
        screen = self.env.reset()
        state = self.rescale_screen(screen)
        state = np.reshape(np.array(Image.fromarray(state).convert('L')), (state.shape[0], state.shape[1], 1))
        state = np.broadcast_to(state, (state.shape[0], state.shape[1], self.frames_stacked))
        return self.transpose_to_torch(state)
    
    def step(self, action, old_state):
        next_screen, reward, done, info = self.env.step(action)
        rescaled_screen = self.rescale_screen(next_screen)
        rescaled_screen = np.reshape(np.array(Image.fromarray(rescaled_screen).convert('L')), (rescaled_screen.shape[0], rescaled_screen.shape[1], 1))
        next_state = torch.cat((old_state[:,1:self.frames_stacked,:,:], self.transpose_to_torch(rescaled_screen)), 1)
        return next_state, reward, done, info
    
    def train(self, training_step_count):
        self.best_evaluation_score = 0
        
        step_count = 0
        episode_count = 0
        
        evaluation_average_scores = []
        next_evaluation_checkpoint = self.steps_between_evaluations

        while step_count < training_step_count:
            episode_count += 1
            
            state = self.reset_env()
            previous_lives = None
            
            episode_reward = 0
            
            while True:                
                step_count += 1
                if step_count > next_evaluation_checkpoint:
                    average_score = self.evaluate(self.evaluation_episodes_count)
                    evaluation_average_scores.append(average_score)
                    next_evaluation_checkpoint += self.steps_between_evaluations
                
                action = self.policy.get_action(state)
                action_is_no_op = self.policy.use_no_op and self.policy.no_op_duration >= self.policy.episode_step_count
                
                next_state, reward, done, info = self.step(action, state)
                
                # If lost life, tell agent he lost life and restart 
                lost_life = previous_lives and (info['ale.lives'] - previous_lives < 0)
                previous_lives = info['ale.lives']

                episode_reward += reward
                reward = torch.tensor([reward], device=self.device)
                
                if done or lost_life:
                    if not action_is_no_op:
                        self.replay_memory.push(state, action, None, reward)
                    
                    print("Episode ", episode_count, " reward is: ", episode_reward)
                    break

                if not action_is_no_op:
                        self.replay_memory.push(state, action, next_state, reward)

                state = next_state
        
        evaluation_index = np.arange(1, len(evaluation_average_scores) + 1)
        plt.plot(evaluation_index, evaluation_average_scores)
        plt.savefig('training_scores.png')
    
    def evaluate(self, episode_count):
        total_reward = 0
        
        for i in range(episode_count):
            state = self.reset_env()

            episode_reward = 0
            
            while True:
                action = self.policy.get_action(state, mode = 'evaluation')
                
                next_state, reward, done, info = self.step(action, state)                
                
                episode_reward += reward
                
                if done:
                    break

                state = next_state
            
            total_reward += episode_reward
        
        average_reward = total_reward / episode_count
        print("Average score is ", average_reward)
        
        if average_reward > self.best_evaluation_score:
            torch.save(self.policy_net.state_dict(), self.weights_path)
            self.best_evaluation_score = average_reward
        return average_reward

if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)
    
    env_name = config['atari']['env_name']
    

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default=env_name, help='Select the environment to run')
    args = parser.parse_args()

    env = gym.make(args.env_id).unwrapped
    env.seed(0)
    
    agent = DQNAgent(env, config)
    
    
    training_step_count = config['train']['training_step_count']
    agent.train(training_step_count)

    # Close the env and write monitor result info to disk
    env.close()