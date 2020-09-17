import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable

import math




# Factorised NoisyLinear layer with bias
# https://github.com/qfettes/DeepRL-Tutorials
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4, factorised_noise=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.factorised_noise = factorised_noise
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        if self.factorised_noise:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        else:
            self.weight_epsilon.copy_(torch.randn((self.out_features, self.in_features)))
            self.bias_epsilon.copy_(torch.randn(self.out_features))

    def forward(self, inp):
        if self.training:
            return F.linear(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inp, self.weight_mu, self.bias_mu)


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
        print(linear_input_size)
        
        self.dense1 = nn.Linear(linear_input_size, self.dense_size) if not self.use_noisy_nets else NoisyLinear(linear_input_size, self.dense_size, self.sigma_init)
        self.dense2 = nn.Linear(self.dense_size, outputs) if not self.use_noisy_nets else NoisyLinear(self.dense_size, outputs, self.sigma_init)
        
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