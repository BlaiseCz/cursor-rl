import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from bots import BaseBot


class DQN(nn.Module):
    def __init__(self, action_size=4):
        super(DQN, self).__init__()
        
        # Add attention mechanism for opponent positions
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        
        # Add opponent-specific processing
        self.opponent_net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Process position (2D)
        self.position_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Process multiple coins information (30D: 10 coins * (distance + direction))
        self.coins_net = nn.Sequential(
            nn.Linear(30, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Process red coins information (15D: 5 coins * (distance + direction))
        self.red_coins_net = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # Process other players information (6D: 2 players * (distance + direction))
        self.players_net = nn.Sequential(
            nn.Linear(6, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Process scalar features (time, score, counts)
        self.scalar_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # Combine all features
        combined_size = 64 + 64 + 64 + 32 + 32  # Updated to reflect new dimensions
        
        # Value stream (state value)
        self.value_stream = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        
        # Advantage stream (action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_size)
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
        coins_features = self.coins_net(observation['coins'])
        red_coins_features = self.red_coins_net(observation['red_coins'])
        
        # Process opponent features with attention
        player_features_raw = self.players_net(observation['other_players'])
        # Reshape for attention (sequence_length, batch_size, embed_dim)
        player_features = player_features_raw.unsqueeze(0)  # Add sequence dimension
        pos_features_expanded = pos_features.unsqueeze(0)  # Add sequence dimension
        
        # Apply attention between player positions and agent's position
        attended_features, _ = self.attention(
            player_features,  # query
            pos_features_expanded,  # key
            player_features  # value
        )
        # Squeeze back to original dimensions
        attended_features = attended_features.squeeze(0)
        
        scalar_input = torch.cat([
            observation['time_left'],
            observation['score'],
            observation['coins_left'],
            observation['red_coins_left']
        ], dim=1)
        scalar_features = self.scalar_net(scalar_input)
        
        # Combine all features, now using attended player features
        combined = torch.cat([
            pos_features,
            coins_features,
            red_coins_features,
            attended_features,  # Use attended features instead of raw player features
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(100000, alpha=0.6)
        self.beta = 0.4
        self.beta_increment = 0.0005
        
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.update_target_every = 2500
        self.steps = 0
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=100, verbose=True
        )
    
    def get_action(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        
        with torch.no_grad():
            # Convert numpy arrays to tensors more efficiently
            obs = {
                'position': torch.from_numpy(observation['position']).float().unsqueeze(0).to(self.device),
                'coins': torch.from_numpy(observation['coins']).float().unsqueeze(0).to(self.device),
                'red_coins': torch.from_numpy(observation['red_coins']).float().unsqueeze(0).to(self.device),
                'other_players': torch.from_numpy(observation['other_players']).float().unsqueeze(0).to(self.device),
                'time_left': torch.from_numpy(np.array([observation['time_left']], dtype=np.float32)).to(self.device),
                'score': torch.from_numpy(np.array([observation['score']], dtype=np.float32)).to(self.device),
                'coins_left': torch.from_numpy(np.array([observation['coins_left']], dtype=np.float32)).to(self.device),
                'red_coins_left': torch.from_numpy(np.array([observation['red_coins_left']], dtype=np.float32)).to(self.device)
            }
            
            q_values = self.model(obs)
            temperature = 0.5
            probs = F.softmax(q_values / temperature, dim=1)
            return torch.multinomial(probs, 1).item()

    def load(self, path):
        """Load a trained model from a file"""
        try:
            # Load the full checkpoint dictionary with weights_only=False since this is our own trusted model
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Extract just the model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            
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

