import numpy as np
from abc import ABC, abstractmethod

class BaseBot(ABC):
    @abstractmethod
    def get_action(self, observation):
        pass

class RandomBot(BaseBot):
    def get_action(self, observation):
        return np.random.randint(0, 4)

class PolicyBot(BaseBot):
    def get_action(self, observation):
        # Get spatial observation
        spatial_obs = observation['spatial_obs']
        
        # Extract bot position (from channel 1 where other players are marked)
        policy_positions = np.where(spatial_obs[1] > 0.5)
        if len(policy_positions[0]) == 0:
            return 0
        
        # Get bot position (center of the marked area)
        bot_y = int(np.mean(policy_positions[0]))
        bot_x = int(np.mean(policy_positions[1]))
        bot_pos = np.array([bot_x, bot_y])
        
        # Get coin positions (from channel 2)
        coin_positions = np.where(spatial_obs[2] > 0.5)
        if len(coin_positions[0]) == 0:
            return 0
            
        # Convert to list of [x, y] coordinates
        coins = np.array([[x, y] for y, x in zip(coin_positions[0], coin_positions[1])])
        
        # Find nearest coin
        distances = np.linalg.norm(coins - bot_pos, axis=1)
        nearest_coin = coins[np.argmin(distances)]
        
        # Simple policy: move towards nearest coin
        dx = nearest_coin[0] - bot_pos[0]
        dy = nearest_coin[1] - bot_pos[1]
        
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3  # right or left
        else:
            return 2 if dy > 0 else 0  # down or up 