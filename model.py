import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self, config, outputs):
        super(DQN, self).__init__()
        
        h = config['atari']['scaled_image_height']
        w = config['atari']['scaled_image_width']
        
        self.num_layers = config['model']['num_layers']
        self.channels   = config['model']['channels']
        self.kernels    = config['model']['kernels']
        self.strides    = config['model']['strides']
        self.dense_size = config['model']['dense_size']
        
        self.input_channels = config['atari']['frames_stacked']
        
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
        self.dense1 = nn.Linear(linear_input_size, self.dense_size)
        self.dense2 = nn.Linear(self.dense_size, outputs)
        #self.softmax = nn.Softmax(outputs)
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x))
        x = F.relu(self.dense1(x.view(x.size(0), -1)))
        x = self.dense2(x)
        return x