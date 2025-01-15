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
        
        # Process position (2D)
        self.position_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Process nearest coin information (3D: distance + direction)
        self.coin_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Process nearest red coin information (3D: distance + direction)
        self.red_coin_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Process other players information (9D: 3 players * (distance + direction))
        self.players_net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # Process scalar features (time, score, counts)
        self.scalar_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Combine all features
        combined_size = 64 + 64 + 64 + 128 + 64  # Sum of all feature sizes
        
        # Value stream (state value)
        self.value_stream = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Advantage stream (action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, action_size)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, observation):
        # Extract features from observation dictionary
        pos_features = self.position_net(observation['position'])
        coin_features = self.coin_net(observation['nearest_coin'])
        red_coin_features = self.red_coin_net(observation['nearest_red_coin'])
        player_features = self.players_net(observation['other_players'])
        
        scalar_input = torch.cat([
            observation['time_left'],
            observation['score'],
            observation['coins_left'],
            observation['red_coins_left']
        ], dim=1)
        scalar_features = self.scalar_net(scalar_input)
        
        # Combine all features
        combined = torch.cat([
            pos_features,
            coin_features,
            red_coin_features,
            player_features,
            scalar_features
        ], dim=1)
        
        # Dueling architecture
        value = self.value_stream(combined)
        advantages = self.advantage_stream(combined)
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals

class RLBot(BaseBot):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(100000, alpha=0.6)
        self.beta = 0.4
        self.beta_increment = 0.001
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_target_every = 1000
        self.steps = 0
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=100, verbose=True
        )
    
    def get_action(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            # Convert numpy arrays to tensors
            obs = {
                'position': torch.FloatTensor(observation['position']).unsqueeze(0).to(self.device),
                'nearest_coin': torch.FloatTensor(observation['nearest_coin']).unsqueeze(0).to(self.device),
                'nearest_red_coin': torch.FloatTensor(observation['nearest_red_coin']).unsqueeze(0).to(self.device),
                'other_players': torch.FloatTensor(observation['other_players']).unsqueeze(0).to(self.device),
                'time_left': torch.FloatTensor([observation['time_left']]).to(self.device),
                'score': torch.FloatTensor([observation['score']]).to(self.device),
                'coins_left': torch.FloatTensor([observation['coins_left']]).to(self.device),
                'red_coins_left': torch.FloatTensor([observation['red_coins_left']]).to(self.device)
            }
            
            q_values = self.model(obs)
            return torch.argmax(q_values).item()

    def load(self, path):
        """Load a trained model from a file"""
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"Successfully loaded model from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, transition):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta):
        if len(self.memory) == 0:
            return None, None, None
        
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

