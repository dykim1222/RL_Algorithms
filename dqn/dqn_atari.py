import gym
import random
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
k = 4
N = 300000  # 100gb
lr = 0.0005
bs = 32
num_episodes = 1000000
render = False
env_name = "Breakout-v0"
gamma = 0.99
num_epoch = 10
print_freq = 10
target_update = 10


class ReplayMemory(object):


    def __init__(self, k, N, bs):
        self.k = k
        self.N = N
        self.bs = bs
        self.memory = deque(maxlen=N)
        self.obs_hist = []


    def get_phi(self, obs):
        obs_copy = np.copy(obs)
        obs_copy = resize(rgb2gray(obs), (110, 84))
        obs = resize(rgb2gray(obs), (110, 84))[17:17+84, :]
        self.obs_hist.append(obs)
        if len(self.obs_hist) == 1:
            obs = np.stack([obs]*self.k)
        elif len(self.obs_hist) < self.k:
            obs = [self.obs_hist[0]] * (self.k - len(self.obs_hist) + 1)
            obs = np.stack(obs + self.obs_hist[-len(self.obs_hist)+1:])
        else:
            obs = np.stack(self.obs_hist[-self.k:])
        return torch.Tensor(obs).unsqueeze(0).to(device)


    def add_memory(self, phi, action, rew, phi_new, done):
        self.memory.append([phi, action, rew, phi_new, done])


    def sample(self, target_model):
        bs = self.bs if len(self.memory) >= self.bs else len(self.memory)
        batch = random.sample(self.memory, bs)
        batch = [*zip(*batch)]

        x_batch = torch.stack(batch[0]).squeeze(1).to(device)
        a_batch = torch.Tensor(batch[1]).reshape(bs, -1).to(device, torch.long)
        r_batch = torch.Tensor(batch[2]).to(device)
        d_batch = 1.0 - torch.Tensor(batch[4]).to(device)
        y_batch = torch.stack(batch[3]).squeeze(1).to(device)

        y_batch = (r_batch + gamma*(target_model(y_batch).max(dim=1)[0])*d_batch).reshape(bs, 1)
        return x_batch, y_batch.detach(), a_batch


class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(9*9*32, 256)
        self.fc2 = nn.Linear(256, num_actions)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.epi_num = 0.0


    def forward(self, phi):
        phi = F.relu(self.bn1(self.conv1(phi)))
        phi = F.relu(self.bn2(self.conv2(phi)))
        phi = self.fc2(F.relu(self.fc1(phi.reshape(phi.shape[0],-1))))
        return phi


    def update(self, buffer, targetQ):
        for _ in range(num_epoch):
            x_batch, y_batch, a_batch = buffer.sample(targetQ)
            out = self.forward(x_batch).gather(1, a_batch)
            loss = F.smooth_l1_loss(out, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            # for param in self.parameters():
            #     param.grad.data.clamp_(-1, 1)
            self.optimizer.step()


    def get_action(self, phi):
        Q = self.forward(phi)
        eps = max(0.1, 0.5-(9e-7)*self.step_num)
        if npr.rand() < eps:
            return npr.choice(len(Q.squeeze()))
        else:
            return Q.argmax().item()

    def save__(self):
        data = {'model':self.state_dict(),
        'optimizer':self.optimizer.state_dict(),
        'step':self.step_num}
        torch.save(data, 'dqn.pt')


def clip_reward(r):
    if r == 0:
        return r
    elif r > 0:
        return 1.0
    else:
        return -1.0


if __name__ == "__main__":
    env = gym.make(env_name)
    buffer = ReplayMemory(k, N, bs)
    model = DQN(env.action_space.n).to(device)
    targetQ = DQN(env.action_space.n).to(device)
    targetQ.load_state_dict(model.state_dict())
    targetQ.eval()
    avg_total_reward = 0.0
    for epi in range(num_episodes):
        epi_total_reward = 0.0
        obs = env.reset()
        phi = buffer.get_phi(obs)
        while True:
            action = model.get_action(phi)
            obs_new, rew, done, _ = env.step(action)
            rew = clip_reward(rew)
            epi_total_reward += rew
            phi_new = buffer.get_phi(obs_new)
            buffer.add_memory(phi.cpu(), action, rew, phi_new.cpu(), float(done))
            if done:
                avg_total_reward += epi_total_reward
                buffer.obs_hist = []
                model.epi_num += 1
                break
            phi = phi_new.clone()

        if len(buffer.memory) > 1000:
            model.update(buffer, targetQ)

        if (epi+1)%print_freq==0:
            print('Episode: %3d \t Avg Return: %.3f'%(epi+1, avg_total_reward/print_freq))
            avg_total_reward = 0.0

        if (epi+1)%target_update==0:
            targetQ.load_state_dict(model.state_dict())
            targetQ.eval()
            model.save__()
    env.close()
