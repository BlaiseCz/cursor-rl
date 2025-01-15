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

def train(episodes=1000):
    env = CoinCollectionEnv(map_size=(300, 300), render_mode=None)
    policy_bot = PolicyBot()
    rl_bot = RLBot()
    
    best_reward = -np.inf
    episode_rewards = []
    
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
            
            # Store transition in memory
            rl_bot.memory.append((
                observation['spatial_obs'],
                observation['time_left'],
                observation['score'],
                actions['rl'],
                reward,
                next_observation['spatial_obs'],
                next_observation['time_left'],
                next_observation['score'],
                done
            ))
            
            # Train the network
            if len(rl_bot.memory) >= rl_bot.batch_size:
                batch = random.sample(rl_bot.memory, rl_bot.batch_size)
                
                # Prepare batch data
                spatial_obs = torch.FloatTensor(np.array([b[0] for b in batch])).to(rl_bot.device)
                time_left = torch.FloatTensor([b[1] for b in batch]).to(rl_bot.device)
                scores = torch.FloatTensor([b[2] for b in batch]).to(rl_bot.device)
                actions = torch.LongTensor(np.array([b[3] for b in batch])).to(rl_bot.device)
                rewards = torch.FloatTensor(np.array([b[4] for b in batch])).to(rl_bot.device)
                next_spatial = torch.FloatTensor(np.array([b[5] for b in batch])).to(rl_bot.device)
                next_time = torch.FloatTensor([b[6] for b in batch]).to(rl_bot.device)
                next_scores = torch.FloatTensor([b[7] for b in batch]).to(rl_bot.device)
                dones = torch.FloatTensor(np.array([b[8] for b in batch])).to(rl_bot.device)
                
                # Get current Q values
                current_q = rl_bot.model(spatial_obs, time_left, scores).gather(1, actions.unsqueeze(1))
                
                # Get next Q values
                next_q = rl_bot.target_model(next_spatial, next_time, next_scores).max(1)[0].detach()
                target_q = rewards + (1 - dones) * rl_bot.gamma * next_q
                
                # Compute loss and optimize
                loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
                rl_bot.optimizer.zero_grad()
                loss.backward()
                rl_bot.optimizer.step()
            
            total_reward += reward
            observation = next_observation
            rl_bot.steps += 1
            
            # Update target network
            if rl_bot.steps % rl_bot.update_target_every == 0:
                rl_bot.target_model.load_state_dict(rl_bot.model.state_dict())
        
        # Decay epsilon
        rl_bot.epsilon = max(rl_bot.epsilon_min, rl_bot.epsilon * rl_bot.epsilon_decay)
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(rl_bot.model.state_dict(), 'best_model.pth')
        
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"\nEpisode {episode}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Epsilon: {rl_bot.epsilon:.3f}")
            print(f"Average Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train RL bot')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of episodes to train (default: 1000)')
    
    args = parser.parse_args()
    train(episodes=args.episodes) 