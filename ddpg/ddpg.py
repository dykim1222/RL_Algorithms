import gym
import random
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

!pip install tensorboardX
from tensorboardX import SummaryWriter

env_name = "Pendulum-v0"
N = 100000
num_episodes = 1000000
num_epoch = 10
print_freq = 20
bs = 32
lr_mu = 1e-4
lr_Q = 1e-3
gamma = 0.99
tau = 0.002
noise_eps = 1e-3

def update_models(Q, mu, Q_target, mu_target, buffer):
    for _ in range(num_epoch):
        x_batch, a_batch, y_batch = buffer.sample(Q_target, mu_target)

        loss_Q = F.smooth_l1_loss(Q(x_batch, a_batch), y_batch)
        loss_mu = - Q(x_batch, mu(x_batch)).mean()
        Q.optimizer.zero_grad()
        loss_Q.backward()
        Q.optimizer.step()
        mu.optimizer.zero_grad()
        loss_mu.backward()
        mu.optimizer.step()

        for p, p_target in zip(Q.parameters(), Q_target.parameters()):
            p_target.data.copy_(tau*p.data+(1-tau)*p_target.data)
        for p, p_target in zip(mu.parameters(), mu_target.parameters()):
            p_target.data.copy_(tau*p.data+(1-tau)*p_target.data)

class Actor(nn.Module):
    def __init__(self, num_obs, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_mu)

    def forward(self, obs):
        obs = torch.Tensor(obs)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = 2*torch.tanh(self.fc3(obs))
        return obs

class Critic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super(Critic, self).__init__()
        self.fc_obs = nn.Linear(num_obs, 128)
        self.fc_act = nn.Linear(num_actions, 128)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)

    def forward(self, obs, act):
        obs = F.relu(self.fc_obs(obs))
        act = F.relu(self.fc_act(act))
        out = torch.cat((obs,act),1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class ReplayBuffer(nn.Module):
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.memory = deque(maxlen=N)

    def add_transition(self, transition):
        self.memory.append(transition)

    def sample(self, Q_target, mu_target):
        batch = random.sample(self.memory, bs)
        batch = [*zip(*batch)]
        x_batch = torch.Tensor(batch[0])
        a_batch = torch.Tensor(batch[1]).reshape(bs,-1)
        r_batch = torch.Tensor(batch[2]).reshape(bs,1)
        d_batch = 1 - torch.Tensor(batch[4]).reshape(bs,1)
        y_batch = torch.Tensor(batch[3])

        y_batch = (r_batch + gamma*d_batch*Q_target(y_batch, mu_target(y_batch))).reshape(bs,1)
        return x_batch, a_batch, y_batch.detach()

class NoiseProcess(object):
    def __init__(self, noise, num_actions):
        self.eps = noise
        self.num_actions = num_actions

    def __call__(self):
        return torch.Tensor(npr.normal(loc=0.0, scale=self.eps, size=self.num_actions))

def main():
    env = gym.make(env_name)
    writer = SummaryWriter()
    S, A = env.observation_space.shape[0], env.action_space.shape[0]
    Q, mu = Critic(S,A), Actor(S,A)
    Q_target, mu_target = Critic(S,A), Actor(S,A)
    Q_target.load_state_dict(Q.state_dict())
    mu_target.load_state_dict(mu.state_dict())
    Q_target.eval()
    mu_target.eval()
    buffer = ReplayBuffer()
    noise = NoiseProcess(noise_eps, A)
    avg_total_reward = 0.0

    for epi in range(num_episodes):
        obs = env.reset()
        epi_total_reward = 0.0

        while True:
            action = (mu(obs) + noise()).detach().numpy()
            obs_new, rew, done, _ = env.step(action)
            buffer.add_transition((obs, action.item(), rew/100.0, obs_new, float(done)))
            epi_total_reward += rew
            if done:
                avg_total_reward += epi_total_reward
                writer.add_scalar('Return', epi_total_reward, epi)
                break
            obs = np.copy(obs_new)

        if len(buffer.memory) > 2000:
            update_models(Q, mu, Q_target, mu_target, buffer)

        if (epi+1)%print_freq==0:
            print('Episode: %3d \t Avg Return: %.3f'%(epi+1, avg_total_reward/print_freq))
            writer.add_scalar('Avg Return over last {} episodes'.format(print_freq), avg_total_reward/print_freq, epi)
            if avg_total_reward/print_freq > -150:
                data = {'Q':Q.state_dict(), 'mu': mu.state_dict()}
                torch.save(data, 'model.pt')
            avg_total_reward = 0.0

    env.close()
    writer.close()

if __name__ == '__main__':
    main()
