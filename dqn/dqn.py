import gym
import random
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# !pip install tensorboardcolab
# from tensorboardcolab import TensorBoardColab
!pip install tensorboardX
from tensorboardX import SummaryWriter

env_name = "CartPole-v1"
N = 100000
num_episodes = 1000000
num_epoch = 20
update_freq = 20
bs = 128
lr = 0.0001
gamma = 0.99


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=N)

    def add_memory(self, obs, action, rew, obs_new, done):
        self.memory.append([obs, action, rew, obs_new, done])

    def sample(self, target_model):
        batch = random.sample(self.memory, bs)
        batch = [*zip(*batch)]
        x_batch = torch.Tensor(batch[0])
        a_batch = torch.Tensor(batch[1]).reshape(bs,-1).to(torch.long)
        r_batch = torch.Tensor(batch[2])
        d_batch = 1 - torch.Tensor(batch[4])
        y_batch = torch.Tensor(batch[3])
        y_batch = (r_batch + gamma*(target_model(y_batch).max(dim=1)[0])*d_batch).reshape(bs, 1)
        return x_batch, y_batch.detach(), a_batch

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 2)
        self.epi_num = 0.0
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def forward(self, obs):
        obs = torch.Tensor(obs)
        return self.fc2(F.relu(self.fc1(obs)))

    def get_action(self, phi):
        Q = self.forward(phi)
        eps = max(0.01, 0.08 - 0.01*(self.epi_num/200))
        if npr.rand() < eps:
            return npr.choice(len(Q.squeeze()))
        else:
            return Q.argmax().item()

    def update(self, buffer, targetQ):
        for _ in range(num_epoch):
            x_batch, y_batch, a_batch = buffer.sample(targetQ)
            out = self.forward(x_batch).gather(1, a_batch)
            loss = F.smooth_l1_loss(out, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def main(run_num):
    # tb = TensorBoardColab()
    writer = SummaryWriter()
    for z in range(run_num):
        env = gym.make(env_name)
        buffer = ReplayMemory()
        model = DQN()
        targetQ = DQN()
        targetQ.load_state_dict(model.state_dict())
        targetQ.eval()
        avg_total_reward = 0.0
        for epi in range(num_episodes):
            obs = env.reset()
            epi_total_reward = 0.0
            while True:
                action = model.get_action(obs)
                obs_new, rew, done, _ = env.step(action)
                epi_total_reward += rew
                buffer.add_memory(obs, action, rew/100.0, obs_new, float(done))
                if done:
                    # tb.save_value('Return', 'return', epi, epi_total_reward)
#                     writer.add_scalar('Return'.format(z), epi_total_reward, epi)
                    writer.add_scalars('Return', {'run_{}'.format(z): epi_total_reward}, epi)
                    avg_total_reward += epi_total_reward
                    model.epi_num += 1
                    break
                obs = np.copy(obs_new)

            if len(buffer.memory) > 1000:
                model.update(buffer, targetQ)

            if (epi+1)%update_freq==0:
                print('Episode: %3d \t Avg Return: %.3f'%(epi+1, avg_total_reward/update_freq))
                # tb.flush_line('return')
                avg_total_reward = 0.0
                targetQ.load_state_dict(model.state_dict())
                targetQ.eval()

        env.close()
        # tb.close()
    writer.close()

if __name__ == "__main__":
    main(1)
