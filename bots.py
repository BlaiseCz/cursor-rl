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
        
        # Get bot position
        bot_positions = np.where(spatial_obs[1] > 0.5)
        if len(bot_positions[0]) == 0:
            return np.random.randint(0, 4)
        
        bot_y = int(np.mean(bot_positions[0]))
        bot_x = int(np.mean(bot_positions[1]))
        
        # Get coin positions
        coin_positions = np.where(spatial_obs[2] > 0.5)
        if len(coin_positions[0]) == 0:
            return np.random.randint(0, 4)
        
        # Find nearest coin
        coins = list(zip(coin_positions[1], coin_positions[0]))  # x, y coordinates
        distances = [(x-bot_x)**2 + (y-bot_y)**2 for x, y in coins]
        nearest_x, nearest_y = coins[np.argmin(distances)]
        
        # Simple movement: move in direction of largest difference
        dx = nearest_x - bot_x
        dy = nearest_y - bot_y
        
        # Move horizontally or vertically based on which distance is larger
        if abs(dx) >= abs(dy):
            if dx > 0:
                return 1  # right
            else:
                return 3  # left
        else:
            if dy > 0:
                return 2  # down
            else:
                return 0  # up 