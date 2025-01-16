import numpy as np
from abc import ABC, abstractmethod


class BaseBot(ABC):
    @abstractmethod
    def get_action(self, observation):
        pass


class PolicyBot(BaseBot):
    def get_action(self, observation):
        # Get bot position
        bot_pos = observation['position'] * 300  # Denormalize position

        # Get information about closest coins
        coins_info = observation['coins'].reshape(-1, 3)  # Reshape to (10, 3) for 10 coins
        red_coins_info = observation['red_coins'].reshape(-1, 3)  # Reshape to (5, 3) for 5 red coins

        # Find the closest regular coin that's not too close to red coins
        best_coin_idx = None
        best_coin_score = float('-inf')

        for i, coin_info in enumerate(coins_info):
            coin_dist = coin_info[0] * 300  # Denormalize distance
            if coin_dist > 290:  # Skip placeholder coins (they have distance 1.0)
                continue

            # Calculate score for this coin based on distance and proximity to red coins
            coin_score = -coin_dist  # Base score is negative distance (closer is better)

            # Check proximity to red coins
            for red_coin in red_coins_info:
                red_dist = red_coin[0] * 300  # Denormalize distance
                if red_dist < 290:  # Only consider real red coins
                    # If red coin is closer than regular coin, reduce score
                    if red_dist < coin_dist:
                        coin_score -= 100  # Heavy penalty
                    # If red coin is nearby, add penalty based on proximity
                    if red_dist < 100:
                        coin_score -= (100 - red_dist)

            if best_coin_idx is None or coin_score > best_coin_score:
                best_coin_idx = i
                best_coin_score = coin_score

        # If no good coin found, just avoid red coins
        if best_coin_idx is None:
            # Find the closest red coin and move away from it
            closest_red = red_coins_info[0]
            if closest_red[0] * 300 < 100:  # If red coin is close
                dx = -closest_red[1]  # Move in opposite direction
                dy = -closest_red[2]
            else:
                # Random movement if no immediate threats
                dx = np.random.uniform(-1, 1)
                dy = np.random.uniform(-1, 1)
        else:
            # Move towards the best coin
            dx = coins_info[best_coin_idx][1]  # Direction x
            dy = coins_info[best_coin_idx][2]  # Direction y

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


class HumanBot(BaseBot):
    def get_action(self, observation):
        # Get bot position
        bot_pos = observation['position'] * 300  # Denormalize position

        # Get information about closest coins
        coins_info = observation['coins'].reshape(-1, 3)  # Reshape to (10, 3) for 10 coins
        red_coins_info = observation['red_coins'].reshape(-1, 3)  # Reshape to (5, 3) for 5 red coins

        # Find the closest coin, ignoring red coins completely
        closest_coin_idx = None
        closest_dist = float('inf')

        for i, coin_info in enumerate(coins_info):
            coin_dist = coin_info[0] * 300  # Denormalize distance
            if coin_dist > 290:  # Skip placeholder coins
                continue
            
            if coin_dist < closest_dist:
                closest_dist = coin_dist
                closest_coin_idx = i

        # If no coins found, move randomly
        if closest_coin_idx is None:
            dx = np.random.uniform(-1, 1)
            dy = np.random.uniform(-1, 1)
        else:
            # Move towards the closest coin
            dx = coins_info[closest_coin_idx][1]  # Direction x
            dy = coins_info[closest_coin_idx][2]  # Direction y

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
