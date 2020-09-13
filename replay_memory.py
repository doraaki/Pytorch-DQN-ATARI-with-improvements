from collections import namedtuple
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity, use_priority_replay, prob_alpha = 0.4, importance_sampling_beta = 0.6):
        self.use_priority_replay = use_priority_replay
        self.prob_alpha = prob_alpha
        self.importance_sampling_beta = importance_sampling_beta
        
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
        self.priorities = np.zeros((capacity, ), dtype = np.float32)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        maximum_priority = self.priorities.max() if self.memory else 1.0
        
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = maximum_priority
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if not self.use_priority_replay:
            return random.sample(self.memory, batch_size)
        
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

            
        probs = np.power(priorities + 1e-6, self.prob_alpha)
        probs /= probs.sum()
        
        #print(probs)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        importance_sampling_weights = (len(self.memory) * probs[indices]) ** (-self.importance_sampling_beta)
        importance_sampling_weights /= importance_sampling_weights.max()
        importance_sampling_weights = np.array(importance_sampling_weights, dtype=np.float32)

        return samples, indices, importance_sampling_weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority
        
        
    def __len__(self):
        return len(self.memory)
    
    def flush(self):
        self.memory.clear()
        self.position = 0