import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import numpy as np
import gym
import matplotlib.pyplot as plt
import pdb
from collections import defaultdict

env_name = "CartPole-v1"
render = False

num_hidden = 128
lr = 0.0001
gamma = 0.99
max_steps = 4000
print_freq = 20
update_freq = 10
beta = 0.01
lam = 0.95

seed=1
np.random.seed(seed)
torch.manual_seed(seed)

def ensure_shared_grads(model, shared_model):
# https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

class ActorCritic(nn.Module):
    def __init__(self, num_input, num_output):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fcv = nn.Linear(num_hidden, 1)
        self.fcv.weight.data.mul_(0.1)
        self.fcv.bias.data.mul_(0.0)
        self.fcp = nn.Linear(num_hidden, num_output)

    def a(self, s):
        s = torch.Tensor(s)
        logits = self.fcp(F.relu(self.fc2(F.relu(self.fc1(s)))))
        prob, logprob = F.softmax(logits, dim=-1), F.log_softmax(logits, dim=-1)
        a = prob.multinomial(1)
        entropy = -prob.dot(logprob)
        return a, logprob[a], entropy

    def v(self, s):
        s = torch.Tensor(s)
        return self.fcv(F.relu(self.fc2(F.relu(self.fc1(s)))))


def train(shared_model, proc_num, num_processes, S, A, logs):

    env = gym.make(env_name)
    env.seed(proc_num)
    model = ActorCritic(S,A)
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=lr)
    avg_score = 0.0
    for epi in range(max_steps//mp.cpu_count()):

        model.load_state_dict(shared_model.state_dict())
        obs = env.reset()
        score = 0.0

        while True:

            r_lst, logprob_lst, entropy_lst, value_lst = [],[],[],[]

            for j in range(update_freq):

                action, logprob, entropy = model.a(obs)
                obs_new, rew, done, _ = env.step(action.item())
                score += rew
                r_lst.append(rew)
                logprob_lst.append(logprob[0])
                entropy_lst.append(entropy)
                value_lst.append(model.v(obs))

                if done:
                    logs[epi*num_processes+proc_num] = score
                    avg_score += score/print_freq
                    break
                obs = obs_new

            # compute label and loss and backward
            R = 0.0 if done else model.v(obs_new).item()
            value_lst.append(R)
            values  = torch.stack(value_lst[:-1]).reshape(-1)
            returns = torch.zeros(len(r_lst))
            advant  = torch.zeros(len(r_lst))

            for i in reversed(range(len(r_lst))):
                returns[i] = r_lst[i] + gamma*(R if i==len(r_lst)-1 else returns[i+1])
                delta      = r_lst[i] + gamma * value_lst[i + 1] - value_lst[i]
                advant[i]  = delta + gamma*lam*(0 if i==len(r_lst)-1 else advant[i+1])

            val_loss = F.mse_loss(values, returns)
            pol_loss = -torch.stack(logprob_lst).dot(advant.detach()) - beta*torch.stack(entropy_lst).sum()

            optimizer.zero_grad()
            (pol_loss + val_loss).backward()
            ensure_shared_grads(model, shared_model)
            optimizer.step()
            if done:
                break
            obs = obs_new

        if (epi+1)%print_freq == 0:
            print('Process: {} \t Episode: {} \t Score: {}'.format(proc_num,epi+1, avg_score))
            avg_score = 0.0

    env.close()

def main():
    logs = mp.Array('d', range(max_steps))
    num_processes = mp.cpu_count()
    print('We got {} workers!!!'.format(num_processes))
    env = gym.make(env_name)
    S, A = env.observation_space.shape[0], env.action_space.n
    shared_model = ActorCritic(S, A).share_memory()

    procs = []
    for i in range(num_processes):
        proc = mp.Process(target=train, args=(shared_model, i, num_processes, S, A, logs))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    env.close()

if __name__ == "__main__":
    main()
