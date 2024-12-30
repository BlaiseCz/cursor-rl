import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from bots import BaseBot

class DQN(nn.Module):
    def __init__(self, action_size=4):
        super(DQN, self).__init__()
        
        # Spatial feature extraction
        self.spatial_net = nn.Sequential(
            # Input: 3x300x300
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # -> 32x74x74
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64x36x36
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),  # -> 64x17x17
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),  # -> 32x8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 32x4x4
            nn.Flatten(),  # -> 512
        )
        
        # Additional features processing
        self.aux_net = nn.Sequential(
            nn.Linear(2, 64),  # time_left and score
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combined decision making
        self.combined_net = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, spatial_obs, time_left, score):
        spatial_features = self.spatial_net(spatial_obs)
        aux_features = self.aux_net(torch.cat([time_left, score], dim=1))
        combined = torch.cat([spatial_features, aux_features], dim=1)
        return self.combined_net(combined)

class RLBot(BaseBot):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 10
        self.steps = 0
    
    def get_action(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            spatial_obs = torch.FloatTensor(observation['spatial_obs']).unsqueeze(0).to(self.device)
            time_left = torch.FloatTensor([observation['time_left']]).to(self.device)
            score = torch.FloatTensor([observation['score']]).to(self.device)
            
            q_values = self.model(spatial_obs, time_left, score)
            return torch.argmax(q_values).item() 