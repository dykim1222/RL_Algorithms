import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
import numpy as np
import gym
import matplotlib.pyplot as plt
import pdb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_hidden = 128
render = False
env_name = "CartPole-v1"
lr = 0.003
gamma = 0.99
num_episodes = 2000
print_freq = 50
episodic_update = True

class ActorCritic(nn.Module):
    def __init__(self, num_input, num_output, ep_update):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
                        nn.Linear(num_input, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, num_output))
        self.critic = nn.Sequential(
                        nn.Linear(num_input, num_hidden),
                        nn.ReLU(),
                        nn.Linear(num_hidden, 1))
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.ep_update = ep_update
        self.losses = []

    def a(self, s):
        s = torch.Tensor(s).to(device)
        s = self.actor(s)
        m = Categorical(logits=s)
        act = m.sample()
        return act, m.log_prob(act)

    def v(self, s):
        s = torch.Tensor(s).to(device)
        return self.critic(s)

    def update(self, loss):
        if self.ep_update:
            self.losses.append(loss)
        else:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def finish_episode(self):
        if len(self.losses) == 0:
            pass
        else:
            loss = torch.stack(self.losses).sum()/len(self.losses)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.losses = []

def train():
    env = gym.make(env_name)
    writer = SummaryWriter('actor_critic_runs')
    dim_state, dim_action = env.observation_space.shape[0], env.action_space.n
    model = ActorCritic(dim_state, dim_action, episodic_update).to(device)
    avg_returns = 0.0
    for episode in range(num_episodes):
        returns = 0
        obs = env.reset()
        while True:
            act, log_prob = model.a(obs)
            obs_new, rew, done, _ = env.step(act.item())
            td_error = rew + gamma*model.v(obs_new) - model.v(obs)
            loss = -td_error.item()*log_prob + td_error*td_error
            model.update(loss)
            obs = obs_new.copy()
            returns += rew
            if done:
                writer.add_scalar('Return', returns, episode)
                avg_returns += returns
                model.finish_episode()
                break
        if episode%print_freq==0 and episode>0:
            print('Episode: %3d \t Avg Return: %.3f'%(episode, avg_returns/print_freq))
            avg_returns = 0.0
    env.close()
    writer.close()

if __name__ == '__main__':
    train()
