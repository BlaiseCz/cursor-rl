from env import CoinCollectionEnv
from bots import PolicyBot
from rl_bot import RLBot
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

def plot_rewards(rewards, window_size=100):
    """Plot rewards with moving average"""
    plt.figure(figsize=(10, 5))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Only plot moving average if we have enough data points
    if len(rewards) >= window_size:
        # Calculate and plot moving average
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                color='red', label=f'Moving Average ({window_size} episodes)')
    
    plt.title('Training Rewards over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('rewards_plot.png')
    plt.close()

def train(episodes=10000):
    env = CoinCollectionEnv(map_size=(300, 300), render_mode=None)
    policy_bot = PolicyBot()
    rl_bot = RLBot()
    
    best_reward = -np.inf
    episode_rewards = []
    
    # Create a window for tracking performance
    reward_window = deque(maxlen=100)
    
    for episode in tqdm(range(episodes)):
        observation, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get actions for both bots
            actions = {
                'policy': policy_bot.get_action(observation),
                'rl': rl_bot.get_action(observation)
            }
            
            # Step environment
            next_observation, reward, done, _, info = env.step(actions)
            
            # Store transition with priority
            transition = (
                {k: v for k, v in observation.items()},  # Current observation dict
                actions['rl'],
                reward,
                {k: v for k, v in next_observation.items()},  # Next observation dict
                done
            )
            rl_bot.memory.push(transition)
            
            # Train the network
            if len(rl_bot.memory.memory) >= rl_bot.batch_size:
                # Sample with priorities
                batch, indices, weights = rl_bot.memory.sample(rl_bot.batch_size, rl_bot.beta)
                
                if batch:
                    # Prepare batch data
                    current_obs = {
                        'position': torch.FloatTensor(np.array([b[0]['position'] for b in batch])).to(rl_bot.device),
                        'nearest_coin': torch.FloatTensor(np.array([b[0]['nearest_coin'] for b in batch])).to(rl_bot.device),
                        'nearest_red_coin': torch.FloatTensor(np.array([b[0]['nearest_red_coin'] for b in batch])).to(rl_bot.device),
                        'other_players': torch.FloatTensor(np.array([b[0]['other_players'] for b in batch])).to(rl_bot.device),
                        'time_left': torch.FloatTensor(np.array([b[0]['time_left'] for b in batch])).to(rl_bot.device),
                        'score': torch.FloatTensor(np.array([b[0]['score'] for b in batch])).to(rl_bot.device),
                        'coins_left': torch.FloatTensor(np.array([b[0]['coins_left'] for b in batch])).to(rl_bot.device),
                        'red_coins_left': torch.FloatTensor(np.array([b[0]['red_coins_left'] for b in batch])).to(rl_bot.device)
                    }
                    
                    next_obs = {
                        'position': torch.FloatTensor(np.array([b[3]['position'] for b in batch])).to(rl_bot.device),
                        'nearest_coin': torch.FloatTensor(np.array([b[3]['nearest_coin'] for b in batch])).to(rl_bot.device),
                        'nearest_red_coin': torch.FloatTensor(np.array([b[3]['nearest_red_coin'] for b in batch])).to(rl_bot.device),
                        'other_players': torch.FloatTensor(np.array([b[3]['other_players'] for b in batch])).to(rl_bot.device),
                        'time_left': torch.FloatTensor(np.array([b[3]['time_left'] for b in batch])).to(rl_bot.device),
                        'score': torch.FloatTensor(np.array([b[3]['score'] for b in batch])).to(rl_bot.device),
                        'coins_left': torch.FloatTensor(np.array([b[3]['coins_left'] for b in batch])).to(rl_bot.device),
                        'red_coins_left': torch.FloatTensor(np.array([b[3]['red_coins_left'] for b in batch])).to(rl_bot.device)
                    }
                    
                    actions = torch.LongTensor(np.array([b[1] for b in batch])).to(rl_bot.device)
                    rewards = torch.FloatTensor(np.array([b[2] for b in batch])).to(rl_bot.device)
                    dones = torch.FloatTensor(np.array([b[4] for b in batch])).to(rl_bot.device)
                    weights = torch.FloatTensor(weights).to(rl_bot.device)
                    
                    # Get current Q values
                    current_q = rl_bot.model(current_obs).gather(1, actions.unsqueeze(1))
                    
                    # Get next Q values with double DQN
                    next_actions = rl_bot.model(next_obs).max(1)[1]
                    next_q = rl_bot.target_model(next_obs)
                    next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze()
                    
                    # Calculate target Q values
                    target_q = rewards + (1 - dones) * rl_bot.gamma * next_q
                    
                    # Calculate loss with importance sampling weights
                    loss = (weights * F.smooth_l1_loss(current_q.squeeze(), target_q, reduction='none')).mean()
                    
                    # Update priorities
                    priorities = abs(target_q - current_q.squeeze()).detach().cpu().numpy()
                    rl_bot.memory.update_priorities(indices, priorities + 1e-6)
                    
                    # Optimize
                    rl_bot.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(rl_bot.model.parameters(), 1.0)
                    rl_bot.optimizer.step()
            
            total_reward += reward
            observation = next_observation
            rl_bot.steps += 1
            
            # Update target network
            if rl_bot.steps % rl_bot.update_target_every == 0:
                rl_bot.target_model.load_state_dict(rl_bot.model.state_dict())
            
            # Update beta for prioritized replay
            rl_bot.beta = min(1.0, rl_bot.beta + rl_bot.beta_increment)
        
        # Decay epsilon
        rl_bot.epsilon = max(rl_bot.epsilon_min, rl_bot.epsilon * rl_bot.epsilon_decay)
        
        # Update reward tracking
        episode_rewards.append(total_reward)
        reward_window.append(total_reward)
        avg_reward = np.mean(reward_window)
        
        # Update learning rate based on performance
        rl_bot.scheduler.step(avg_reward)
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save({
                'model_state_dict': rl_bot.model.state_dict(),
                'optimizer_state_dict': rl_bot.optimizer.state_dict(),
                'episode': episode,
                'best_reward': best_reward
            }, 'best_model.pth')
        
        if episode % 10 == 0:
            print(f"\nEpisode {episode}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"Epsilon: {rl_bot.epsilon:.3f}")
            print(f"Beta: {rl_bot.beta:.3f}")
            
            # Plot rewards
            plot_rewards(episode_rewards)

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train RL bot')
    parser.add_argument('--episodes', type=int, default=10000,
                      help='Number of episodes to train (default: 10000)')
    
    args = parser.parse_args()
    train(episodes=args.episodes) 