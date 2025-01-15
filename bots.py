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
        # Get bot position and nearest coin information
        bot_pos = observation['position'] * 300  # Denormalize position
        nearest_coin_dist = observation['nearest_coin'][0] * 300  # Denormalize distance
        nearest_coin_dir = observation['nearest_coin'][1:3]  # Get direction vector
        
        # Get nearest red coin information
        nearest_red_dist = observation['nearest_red_coin'][0] * 300  # Denormalize distance
        nearest_red_dir = observation['nearest_red_coin'][1:3]  # Get direction vector
        
        # If there's a red coin too close, move away from it
        if nearest_red_dist < 50:  # Threshold for avoidance
            # Move in opposite direction of red coin
            dx = -nearest_red_dir[0]
            dy = -nearest_red_dir[1]
        else:
            # Move towards nearest regular coin
            dx = nearest_coin_dir[0]
            dy = nearest_coin_dir[1]
        
        # Choose action based on the largest direction component
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