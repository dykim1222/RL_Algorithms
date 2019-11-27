import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from collections import namedtuple

debug = False
env_name = "CartPole-v1"
render = False

num_hidden = 64
lr = 0.0007
actor_loss_weight = 0.1
gamma = 0.99
lam = 0.95
max_steps = 700000
print_freq = 100
update_freq = 400 # n-step update
eps_clip = 0.2
eps = 1e-10
num_epochs = 4
num_processes = 2

seed=1
np.random.seed(seed)
torch.manual_seed(seed)

Transition = namedtuple('Transition',('state', 'action', 'prob', 'mask', 'reward', 'state_next'))

class ActorCritic(nn.Module):
    def __init__(self, num_input, num_output):
        super(ActorCritic, self).__init__()
        self.num_input = num_input
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fcp  = nn.Linear(num_hidden, num_output)
        self.fcv = nn.Linear(num_hidden, 1)
        self.fcv.weight.data.mul_(0.1)
        self.fcv.bias.data.mul_(0.0)

    def pi(self, s):
        s = torch.Tensor(s).reshape(-1, self.num_input)
        s = self.fcp(F.relu(self.fc2(F.relu(self.fc1(s)))))
        return F.softmax(s, dim=1).clamp(max=1 - eps**2)

    def v(self, s):
        s = torch.Tensor(s).reshape(-1, self.num_input)
        return self.fcv(F.relu(self.fc2(F.relu(self.fc1(s)))))

def ensure_shared_grads(model, shared_model):
# https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(shared_model, proc_num, num_processes, S, A, logs):

    env = gym.make(env_name)
    env.seed(proc_num)
    model = ActorCritic(S,A)
    model.train()
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=lr)

    score = 0.0
    avg_score = 0.0
    step = 0
    epi = 0
    obs = env.reset()

    while step < (max_steps//num_processes):

        '''interact'''
        model.load_state_dict(shared_model.state_dict())
        traj_lst = []

        with torch.no_grad():

            for j in range(update_freq):

                prob = model.pi(obs)
                action = prob.multinomial(1)
                obs_new, rew, done, _ = env.step(action.item())
                score += rew
                mask = 0.0 if done else 1.0
                step += 1
                traj_lst.append((obs, np.array([action]), prob, mask, rew, obs_new))

                if done:
                    epi += 1
                    logs[epi*num_processes+proc_num] = score
                    avg_score += score/print_freq
                    score = 0.0

                    if (epi%print_freq == 0) and (epi > 0) :
                        print('Process: {} \t Episode: {} \t Step: {} \t Score: {:6.1f}'.format(proc_num, epi, step, avg_score))
                        avg_score = 0.0
                    obs = env.reset()

                else:
                    obs = obs_new

        batch = Transition(*zip(*traj_lst))
        states = torch.Tensor(batch.state)
        state_last = torch.Tensor(batch.state_next)[-1]
        actions = torch.Tensor(batch.action).long()
        mus = torch.stack(batch.prob).squeeze(1)
        masks = torch.Tensor(batch.mask).unsqueeze(1)
        rews = torch.Tensor(batch.reward).unsqueeze(1)

        '''compute gae and optimize for multiple epochs'''
        for _ in range(num_epochs):

            values = model.v(states)
            value_last = model.v(state_last)

            rets = torch.zeros_like(values)
            L = values.shape[0]
            for t in reversed(range(L)):
                rets[t] = rews[t] + gamma*masks[t]*(value_last[-1] if t==L-1 else rets[t+1])

            rets = (rets- rets.mean())/(rets.std() + eps)
            advs = rets - values
            advs = (advs - advs.mean())/(advs.std()+eps)

            pi = model.pi(states)
            ratio = torch.exp((pi+eps).log() - (mus+eps).log()).gather(1,actions)

            loss1 = ratio*advs
            loss2 = ratio.clamp(min=1-eps_clip, max=1+eps_clip)*advs
            loss_clip = -actor_loss_weight*torch.min(loss1, loss2).mean()
            loss_v = F.smooth_l1_loss(values, rets)

            optimizer.zero_grad()
            (loss_clip + loss_v).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 40.0)
            ensure_shared_grads(model, shared_model) # transfer gradients
            optimizer.step()

    env.close()

def main():

    logs = mp.Array('d', range(5000))
    print('We got {} workers!!!'.format(num_processes))
    env = gym.make(env_name)
    S, A = env.observation_space.shape[0], env.action_space.n
    env.close()

    shared_model = ActorCritic(S,A).share_memory()

    if debug:
        train(shared_model, 0, 1, S, A, logs)

    else:
        procs = []
        for i in range(num_processes):
            proc = mp.Process(target=train, args=(shared_model, i, num_processes, S, A, logs))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

if __name__ == "__main__":
    main()
