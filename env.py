import gym
from gym import spaces
import numpy as np
import time
import pygame

class CoinCollectionEnv(gym.Env):
    """
    A 2D environment where agents compete to collect coins.
    """
    
    PLAYER_COLORS = {
        'human': (0, 0, 255),    # Blue
        'policy': (255, 0, 0),   # Red
        'random': (0, 255, 0),   # Green
        'rl': (255, 165, 0)      # Orange
    }
    
    COIN_COLOR = (255, 255, 0)   # Yellow
    PLAYER_SPEED = 5
    PLAYER_RADIUS = 10
    COIN_RADIUS = 5
    COLLECTION_RADIUS = 20
    
    def __init__(self, map_size=(300, 300), render_mode=None, num_coins=20):
        super().__init__()
        
        if map_size != (300, 300):
            raise ValueError("Map size must be 300x300 for RL compatibility")
        
        # Environment parameters
        self.map_size = map_size
        self.render_mode = render_mode
        self.num_coins = num_coins
        self.game_duration = 15  # seconds
        self.max_steps = 300
        
        # Initialize spaces
        self._init_spaces()
        
        # Pygame setup for rendering
        self._init_pygame() if render_mode == "human" else None
        
        # Add coin spawn timer
        self.last_coin_spawn = 0
        self.coin_spawn_interval = 1  # seconds
        self.coins_per_spawn = 3
        
        # Reset environment
        self.reset()
    
    def _init_spaces(self):
        """Initialize observation and action spaces"""
        # Action space: 0: up, 1: right, 2: down, 3: left
        self.action_space = spaces.Dict({
            'human': spaces.Discrete(4),
            'policy': spaces.Discrete(4),
            'random': spaces.Discrete(4),
            'rl': spaces.Discrete(4)
        })
        
        # Observation space
        self.observation_space = spaces.Dict({
            'spatial_obs': spaces.Box(low=0, high=1, shape=(3, 300, 300), dtype=np.float32),
            'time_left': spaces.Box(low=0, high=self.game_duration, shape=(1,), dtype=np.float32),
            'score': spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32)
        })
    
    def _init_pygame(self):
        """Initialize Pygame for rendering"""
        pygame.init()
        pygame.display.init()
        
        # Create window with extra space for scoreboard
        self.window = pygame.display.set_mode((self.map_size[0] + 200, self.map_size[1]))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Set window title
        pygame.display.set_caption("Coin Collection Game")
    
    def reset(self, seed=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset game state
        self.step_count = 0
        self.start_time = time.time()
        self.last_coin_spawn = self.start_time  # Reset coin spawn timer
        
        # Generate game elements
        self._generate_players()
        self._generate_coins()
        
        # Initialize scores
        self.player_scores = {player: 0 for player in self.PLAYER_COLORS.keys()}
        
        return self._get_obs(), self._get_info()
    
    def _generate_players(self):
        """Generate initial player positions"""
        self.player_positions = {}
        margin = 50
        
        for player in self.PLAYER_COLORS.keys():
            while True:
                pos = self.np_random.uniform(
                    low=[margin, margin],
                    high=[self.map_size[0]-margin, self.map_size[1]-margin],
                    size=(2,)
                )
                # Ensure players don't spawn too close to each other
                if not any(np.linalg.norm(pos - p) < margin for p in self.player_positions.values()):
                    self.player_positions[player] = pos
                    break
    
    def _generate_coins(self):
        """Generate coin positions"""
        margin = 30
        coin_positions = []
        
        for _ in range(self.num_coins):
            while True:
                pos = self.np_random.uniform(
                    low=[margin, margin],
                    high=[self.map_size[0]-margin, self.map_size[1]-margin],
                    size=(2,)
                )
                # Ensure coins don't spawn too close to each other
                if not any(np.linalg.norm(pos - np.array(coin)) < margin for coin in coin_positions):
                    coin_positions.append(pos)
                    break
        
        self.coins = np.array(coin_positions) if coin_positions else np.zeros((0, 2))
    
    def step(self, actions):
        """Execute one time step within the environment"""
        self.step_count += 1
        
        # Store previous state for reward calculation
        previous_positions = {k: v.copy() for k, v in self.player_positions.items()}
        previous_scores = self.player_scores.copy()
        
        # Process actions
        for player, action in actions.items():
            self._move_player(player, action)
        
        # Check if it's time to spawn new coins
        current_time = time.time()
        if current_time - self.last_coin_spawn >= self.coin_spawn_interval:
            self._spawn_new_coins()
            self.last_coin_spawn = current_time
        
        # Update game state
        self._check_coin_collisions()
        
        # Check if episode is done
        done = self._is_done()
        
        # Calculate reward
        reward = self._calculate_reward(previous_positions, previous_scores)
        
        return self._get_obs(), reward, done, False, self._get_info()
    
    def _move_player(self, player_name, action):
        """Move player based on action"""
        if action is None:
            return
            
        # Movement vectors for each action
        moves = {
            0: np.array([0, -1]),  # Up
            1: np.array([1, 0]),   # Right
            2: np.array([0, 1]),   # Down
            3: np.array([-1, 0])   # Left
        }
        
        # Update position
        new_pos = self.player_positions[player_name] + moves[action] * self.PLAYER_SPEED
        
        # Clip to map boundaries
        new_pos[0] = np.clip(new_pos[0], 0, self.map_size[0])
        new_pos[1] = np.clip(new_pos[1], 0, self.map_size[1])
        
        self.player_positions[player_name] = new_pos
    
    def _check_coin_collisions(self):
        """Check for collisions between players and coins"""
        if not hasattr(self, 'coins') or len(self.coins.shape) != 2 or self.coins.shape[0] == 0:
            return
        
        for player_name, player_pos in self.player_positions.items():
            distances = np.linalg.norm(self.coins - player_pos, axis=1)
            collected = distances < self.COLLECTION_RADIUS
            
            if np.any(collected):
                # Update score
                self.player_scores[player_name] += np.sum(collected)
                
                # Remove collected coins
                self.coins = self.coins[~collected]
    
    def _is_done(self):
        """Check if episode is done"""
        if self.render_mode == "human":
            return time.time() - self.start_time >= self.game_duration
        return self.step_count >= self.max_steps or len(self.coins) == 0
    
    def _calculate_reward(self, previous_positions, previous_scores):
        """Calculate reward for RL agent"""
        rl_pos = self.player_positions['rl']
        reward = 0.0
        
        # Coin collection reward
        if previous_scores['rl'] < self.player_scores['rl']:
            reward += 1.0
        
        # Distance to nearest coin reward
        if len(self.coins) > 0:
            distances = np.linalg.norm(self.coins - rl_pos, axis=1)
            min_distance = np.min(distances)
            prev_distances = np.linalg.norm(self.coins - previous_positions['rl'], axis=1)
            prev_min_distance = np.min(prev_distances)
            reward += (prev_min_distance - min_distance) / 100.0
        
        # Movement penalty
        if np.array_equal(rl_pos, previous_positions['rl']):
            reward -= 0.01
        
        return reward
    
    def _get_obs(self):
        """Get current observation"""
        # Create a 3-channel 300x300 observation space
        obs_array = np.zeros((3, 300, 300), dtype=np.float32)
        
        # Channel 0: RL bot position (value 1.0)
        rl_pos = self.player_positions['rl'].astype(int)
        obs_array[0, 
                 max(0, rl_pos[1]-5):min(300, rl_pos[1]+6), 
                 max(0, rl_pos[0]-5):min(300, rl_pos[0]+6)] = 1.0
        
        # Channel 1: Other players (value 1.0)
        for name, pos in self.player_positions.items():
            if name != 'rl':
                pos = pos.astype(int)
                obs_array[1,
                         max(0, pos[1]-5):min(300, pos[1]+6),
                         max(0, pos[0]-5):min(300, pos[0]+6)] = 1.0
        
        # Channel 2: Coins (value 1.0)
        for coin in self.coins:
            coin_pos = coin.astype(int)
            obs_array[2,
                     max(0, coin_pos[1]-3):min(300, coin_pos[1]+4),
                     max(0, coin_pos[0]-3):min(300, coin_pos[0]+4)] = 1.0
        
        return {
            'spatial_obs': obs_array,  # 3-channel spatial observation
            'time_left': np.array([self.game_duration - (time.time() - self.start_time)], dtype=np.float32),
            'score': self.player_scores['rl']
        }
    
    def _get_info(self):
        """Get additional information"""
        return {
            'scores': self.player_scores,
            'coins_left': len(self.coins),
            'step_count': self.step_count
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode != "human":
            return
            
        # Clear screen
        self.window.fill((255, 255, 255))
        
        # Draw coins
        for coin_pos in self.coins:
            pygame.draw.circle(self.window, self.COIN_COLOR, coin_pos.astype(int), self.COIN_RADIUS)
        
        # Draw players
        for player, pos in self.player_positions.items():
            pygame.draw.circle(self.window, self.PLAYER_COLORS[player], 
                            pos.astype(int), self.PLAYER_RADIUS)
        
        # Draw scores and time
        self._render_scoreboard()
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def _render_scoreboard(self):
        """Render scoreboard"""
        y_offset = 20
        time_left = max(0, self.game_duration - (time.time() - self.start_time))
        time_text = self.font.render(f"Time: {time_left:.1f}", True, (0, 0, 0))
        self.window.blit(time_text, (self.map_size[0] + 10, y_offset))
        y_offset += 40
        
        for player, score in self.player_scores.items():
            score_text = self.font.render(f"{player}: {score}", True, self.PLAYER_COLORS[player])
            self.window.blit(score_text, (self.map_size[0] + 10, y_offset))
            y_offset += 40
        
        # Draw separator line
        pygame.draw.line(self.window, (0, 0, 0), 
                        (self.map_size[0], 0), 
                        (self.map_size[0], self.map_size[1]), 2)
    
    def close(self):
        """Clean up resources"""
        if self.render_mode == "human":
            pygame.quit() 
    
    def _spawn_new_coins(self):
        """Spawn new coins on the map"""
        margin = 30
        new_coins = []
        
        for _ in range(self.coins_per_spawn):
            spawn_attempts = 0
            max_attempts = 50  # Prevent infinite loop
            
            while spawn_attempts < max_attempts:
                pos = self.np_random.uniform(
                    low=[margin, margin],
                    high=[self.map_size[0]-margin, self.map_size[1]-margin],
                    size=(2,)
                )
                # Check against existing coins
                existing_coins = self.coins if len(self.coins.shape) == 2 else np.zeros((0, 2))
                if not any(np.linalg.norm(pos - coin) < margin for coin in existing_coins):
                    new_coins.append(pos)
                    break
                spawn_attempts += 1
        
        if new_coins:
            if len(self.coins.shape) == 2 and self.coins.shape[0] > 0:
                self.coins = np.vstack([self.coins, np.array(new_coins)])
            else:
                self.coins = np.array(new_coins) 