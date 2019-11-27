import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
import numpy as np
import gym
import matplotlib.pyplot as plt
import pdb

num_hidden = 128
render = False
env_name = "CartPole-v1"
lr = 1e-3
gamma = 0.99
num_episodes = 10000
print_freq = 50

class Policy(nn.Module):
    def __init__(self, num_input, num_output):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.m = None
    def get_action(self, x):
        x = torch.Tensor(x).to(device)
        x = self.fc2(F.relu(self.fc1(x)))
        self.m = Categorical(logits=x)
        return self.m.sample()

def train():
    env = gym.make(env_name)
    writer = SummaryWriter('reinforce_runs')
    dim_state, dim_action = env.observation_space.shape[0], env.action_space.n
    policy = Policy(dim_state, dim_action).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    avg_returns = 0.0
    
    for episode in range(num_episodes):
        
        log_probs, rewards = [], []
        obs = env.reset()
        
        while True:
            if render: env.render()
            action = policy.get_action(obs)
            obs, rew, done, _ = env.step(action.item())
            log_probs.append(policy.m.log_prob(action))
            rewards.append(rew)
            
            if done:
                writer.add_scalar('Return', np.sum(rewards), episode)
                avg_returns += np.sum(rewards)
                
                for i in reversed(range(len(rewards))):
                    rewards[i] = rewards[i] + gamma*(rewards[i+1] if i<len(rewards)-1 else 0)
                rewards = torch.Tensor(rewards)
                rewards = (rewards - rewards.mean())/rewards.std()
                loss = -torch.stack(log_probs).dot(torch.Tensor(rewards))
                break
                
        if episode%print_freq==0 and episode>0:
            print('Episode: %3d \t Avg Return: %.3f'%(episode, avg_returns/print_freq))
            avg_returns = 0.0
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    env.close()
    writer.close()

if __name__ == '__main__':
    train()
