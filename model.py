import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable

import math


# Noisy linear layer with independent Gaussian noise
# https://github.com/Kaixhin/NoisyNet-A3C
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant(self.sigma_weight, self.sigma_init)
      init.constant(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * torch.tensor(self.epsilon_weight, device = self.device), self.bias + self.sigma_bias * torch.tensor(self.epsilon_bias, device = self.device))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)


class DQN(nn.Module):
    def __init__(self, config, outputs):
        super(DQN, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        h = config['atari']['scaled_image_height']
        w = config['atari']['scaled_image_width']
        
        self.num_layers = config['model']['num_layers']
        self.channels   = config['model']['channels']
        self.kernels    = config['model']['kernels']
        self.strides    = config['model']['strides']
        self.dense_size = config['model']['dense_size']
        
        self.input_channels = config['atari']['frames_stacked']
        
        self.use_noisy_nets = config['rl_params']['use_noisy_nets']
        self.sigma_init = config['rl_params']['sigma_init']
        
        self.convs = []
        self.conv1 = nn.Conv2d(self.input_channels, self.channels[0], kernel_size=self.kernels[0], stride=self.strides[0])
        self.convs.append(self.conv1)
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=self.kernels[1], stride=self.strides[1])
        self.convs.append(self.conv2)
        self.conv3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=self.kernels[2], stride=self.strides[2])
        self.convs.append(self.conv3)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        for i in range(self.num_layers):
            w = conv2d_size_out(w, self.kernels[i], self.strides[i])
            h = conv2d_size_out(h, self.kernels[i], self.strides[i])
            
        linear_input_size = h * w * self.channels[self.num_layers - 1]
        if self.use_noisy_nets:
            self.dense1 = NoisyLinear(linear_input_size, self.dense_size, sigma_init = self.sigma_init).to(self.device)
            self.dense2 = NoisyLinear(self.dense_size, outputs, sigma_init = self.sigma_init).to(self.device)
        else:
            self.dense1 = nn.Linear(linear_input_size, self.dense_size)
            self.dense2 = nn.Linear(self.dense_size, outputs)
        #self.softmax = nn.Softmax(outputs)
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x))
        x = F.relu(self.dense1(x.view(x.size(0), -1)))
        x = self.dense2(x)
        return x
    
    def sample_noise(self):
        if self.use_noisy_nets:
            self.dense1.sample_noise()
            self.dense2.sample_noise()
    
    def remove_noise(self):
        if self.use_noisy_nets:
            self.dense1.remove_noise()
            self.dense2.remove_noise()