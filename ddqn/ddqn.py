import gym
import random
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

env_name = "CartPole-v1"
rendering = False
N = 100000
num_episodes = 4000
num_epoch = 1
update_freq = 10
bs = 256
lr = 0.0007
gamma = 0.99
seed = 1

class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=N)

    def add_memory(self, obs, action, rew, obs_new, mask):
        self.memory.append([obs, action, rew, obs_new, mask])

    def sample(self):
        batch = random.sample(self.memory, bs)
        return list(map(lambda x: torch.Tensor(x), [*zip(*batch)]))

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, 2)
        self.epi_num = 0.0
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def forward(self, obs):
        obs = torch.Tensor(obs).reshape(-1,4)
        obs = self.fc3(F.relu(self.fc2(F.relu(self.fc1(obs)))))
        return obs

    def get_action(self, phi):
        Q = self.forward(phi)
        eps = max(0.01, 0.08 - 0.01*(self.epi_num/200))
        if npr.rand() < eps:
            return npr.choice(len(Q.squeeze()))
        else:
            return Q.argmax().item()

    def update(self, buffer, targetQ):

        for _ in range(num_epoch):

            obs, action, rew, obs_next, mask = buffer.sample()
            action = action.long()
            # pdb.set_trace()
            q_next = self.forward(obs_next)
            a_max_next = q_next.argmax(dim=1, keepdim=True)

            q_targ = targetQ(obs_next)
            q_targ_a_max = q_targ.gather(1,a_max_next)
            y = (rew + gamma*mask*q_targ_a_max).detach()

            q = self.forward(obs)
            q_a = q.gather(1,action)

            self.optimizer.zero_grad()
            loss = F.smooth_l1_loss(q_a, y)
            loss.backward()
            self.optimizer.step()

    def turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False


def main():

    writer = SummaryWriter()
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = DQN()
    targetQ = DQN()
    targetQ.load_state_dict(model.state_dict())
    targetQ.turn_off_grad()
    buffer = ReplayMemory()

    avg_total_reward = 0.0

    for epi in range(num_episodes):
        obs = env.reset()
        epi_total_reward = 0.0
        while True:
            if rendering: env.render()
            action = model.get_action(obs)
            obs_new, rew, done, _ = env.step(action)
            mask = 0.0 if done else 1.0
            epi_total_reward += rew
            buffer.add_memory(obs, [action], [rew/100.0], obs_new, [mask])
            if done:
                writer.add_scalar('Return', epi_total_reward, epi)
                avg_total_reward += epi_total_reward
                model.epi_num += 1
                break
            obs = np.copy(obs_new)

        if len(buffer.memory) > 1000:
            model.update(buffer, targetQ)

        if (epi+1)%update_freq==0:
            print('Episode: %3d \t Avg Return: %.3f'%(epi+1, avg_total_reward/update_freq))
            avg_total_reward = 0.0
            targetQ.load_state_dict(model.state_dict())
            targetQ.turn_off_grad()

    env.close()
    writer.close()

if __name__ == "__main__":
    main()
