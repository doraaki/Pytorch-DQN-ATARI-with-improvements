import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from replay_memory import Transition


class Policy:
    def __init__(self, config, num_actions, policy_net, target_net, device, replay_memory):
        self.num_actions = num_actions
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        self.replay_memory = replay_memory
        
        rl_config = config['rl_params']
        self.gamma = rl_config['gamma']
        self.use_no_op = rl_config['use_no_op']
        self.max_no_op_duration = rl_config['max_no_op_duration']
        self.final_training_epsilon = rl_config['final_training_epsilon']
        self.epsilon_decay_step_duration = rl_config['epsilon_decay_step_duration']
        self.evaluation_epsilon = rl_config['evaluation_epsilon']
        self.frames_between_ddqn_copy = rl_config['frames_between_ddqn_copy']

        train_config = config['train']
        
        if train_config['loss'] == 'huber':
            self.loss = nn.SmoothL1Loss()
        elif train_config['loss'] == 'MSE':
            self.loss = nn.MSELoss()
        else:
            print('Invalid loss')
            exit(1)
        
        self.learning_rate = train_config['learning_rate']
        self.momentum = train_config['momentum']
        
        if train_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.learning_rate, betas=(self.momentum, 0.999))
        elif train_config['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr = self.learning_rate, momentum = self.momentum)
        else:
            print('Invalid optimizer')
            exit(1)

        self.batch_size = train_config['batch_size']
        
        self.training_step_count = train_config['training_step_count']
        self.warmup_step_count = train_config['warmup_step_count']
        self.steps_between_batches = train_config['steps_between_batches']
        self.clamp_grads = train_config['clamp_grads']
        self.clamp_rewards = train_config['clamp_rewards']

        self.epsilon_decay_between_actions = (1 - self.final_training_epsilon) / self.epsilon_decay_step_duration
        self.step_count = 0
        self.episode_step_count = 0
        self.training_epsilon = 1


    def set_parameters_for_new_episode(self):
        if self.use_no_op:
            self.no_op_duration = random.randint(0, self.max_no_op_duration)
        
        self.episode_step_count = 0
    

    def get_action(self, state, mode = 'training'):
        # Increase step counters
        if mode == 'training':
            self.step_count += 1
        self.episode_step_count += 1
        
        # Optimize model
        if mode == 'training':
            if self.step_count > self.warmup_step_count and self.step_count % self.steps_between_batches == 0:
                self.optimize_policy_net()
        
        # Frameskip is 4
        if self.step_count % (self.frames_between_ddqn_copy / 4) == 0:
            print('Copying policy network to target network')
            self.target_net.load_state_dict(self.policy_net.state_dict())


        if self.use_no_op and self.episode_step_count <= self.no_op_duration:
            return 0

        if mode == 'training':
            epsilon = self.training_epsilon
            self.training_epsilon -= self.epsilon_decay_between_actions
        elif mode == 'evaluation':
            epsilon = self.evaluation_epsilon

        if np.random.uniform() < epsilon:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
            

    def optimize_policy_net(self):
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        
        reward_batch = torch.cat(batch.reward)
        if self.clamp_rewards:
            reward_batch = torch.clamp(reward_batch, -1, 1)

        # Compute Q(s_t, a) - 
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    
        # Compute Q(s_{t+1}) * gamma + reward
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    
        # Compute loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.clamp_grads:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()