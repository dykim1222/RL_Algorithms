import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque

debug = False
env_name = "CartPole-v1"
render = False

num_hidden = 32
buffer_size = 25
batch_size = 4
lr = 0.0005
gamma = 0.99
max_steps = 2000000
print_freq = 100
update_freq = 20
beta = 0.001
c = 10
max_kl = 1
alpha = 0.99
replay_ratio = 4
trust_region = True
actor_loss_weight = 0.1
replay_start = 100
eps = 1e-10

seed=1
np.random.seed(seed)
torch.manual_seed(seed)

def ensure_shared_grads(model, shared_model):
# https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

Transition = namedtuple('Transition',('state', 'action', 'prob', 'mask', 'reward', 'state_next'))

class Memory(object):
    def __init__(self):
        self.memory = deque(maxlen=buffer_size)

    def push(self, traj_lst):
        self.memory.append(Transition(*zip(*traj_lst)))

    def sample(self):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ActorCritic(nn.Module):
    def __init__(self, num_input, num_output):
        super(ActorCritic, self).__init__()
        self.num_input = num_input
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fcp  = nn.Linear(num_hidden, num_output)
        self.fcq = nn.Linear(num_hidden, num_output)
        self.fcq.weight.data.mul_(0.1)
        self.fcq.bias.data.mul_(0.0)

    def pi(self, s):
        s = torch.Tensor(s).reshape(-1, self.num_input)
        s = self.fcp(F.relu(self.fc2(F.relu(self.fc1(s)))))
        return F.softmax(s, dim=1).clamp(max=1 - eps**2)

    def q(self, s):
        s = torch.Tensor(s).reshape(-1, self.num_input)
        return self.fcq(F.relu(self.fc2(F.relu(self.fc1(s)))))

def flatten(item):
    return torch.cat([i.view(-1) for i in item]).contiguous()

def update_params(avg_pol, pol):
    param_pol = flatten(pol.parameters())
    param_avg = flatten(avg_pol.parameters())
    param_avg = alpha*param_avg + (1-alpha)*param_pol

    idx = 0
    for param in avg_pol.parameters():
        length = int(np.prod(list(param.shape)))
        param.data.copy_(param_avg[idx:idx + length].view(param.size()))
        idx += length

def train(average_policy, shared_model, proc_num, num_processes, S, A, logs):

    env = gym.make(env_name)
    env.seed(proc_num)
    memory = Memory()
    model = ActorCritic(S,A)
    model.train()
    optimizer = torch.optim.RMSprop(shared_model.parameters(), lr=lr)

    score = 0.0
    avg_score = 0.0
    step = 0
    epi = 0
    obs = env.reset()

    while step < (max_steps//mp.cpu_count()):

        def ACER(on_policy = False):

            nonlocal step, epi, score, avg_score
            nonlocal obs, optimizer, shared_model
            model.load_state_dict(shared_model.state_dict())

            if on_policy:

                traj_lst = []
                with torch.no_grad():
                    for j in range(update_freq):

                        prob = model.pi(obs)
                        action = prob.multinomial(1)
                        obs_new, rew, done, _ = env.step(action.item())
                        rew = min(1, max(-1, rew)) # clipping reward
                        mask = 0.0 if done else 1.0

                        traj_lst.append((obs, np.array([action]), prob, mask, rew, obs_new))
                        step += 1
                        score += rew

                        if done:

                            epi += 1
                            logs[epi*num_processes+proc_num] = score
                            avg_score += score/print_freq
                            score = 0.0

                            if (epi%print_freq == 0) and (epi > 0) :

                                print('Process: {} \t Episode: {} \t Step: {} \t Score: {:6.1f}'.format(proc_num, epi, step, avg_score))
                                avg_score = 0.0

                            obs = env.reset()
                            break

                        else:
                            obs = obs_new

                memory.push(traj_lst)
                batchs = [memory.memory[-1]]

            else:
                batchs = memory.sample()

            for batch in batchs:

                states = torch.Tensor(batch.state)
                state_next = torch.Tensor(batch.state_next)[-1]
                actions = torch.Tensor(batch.action).long()
                mus = torch.stack(batch.prob).squeeze(1)
                masks = torch.Tensor(batch.mask)
                rews = torch.Tensor(batch.reward)

                f = model.pi(states)
                logf = f.log()
                f_a = average_policy.pi(states).data

                q = model.q(states)
                v = (f*q).sum(1, keepdim=True).data
                rho = f.data/mus
                rho_bar = rho.clamp(max=1)
                TIS = rho.clamp(max=c)
                BC = (1-c/(rho+eps)).clamp(min=0)

                q_ret = (model.pi(state_next)*(model.q(state_next))).sum().item()
                q_ret_lst = []
                for t in reversed(range(len(rews))):
                    q_ret = rews[t] + gamma*q_ret*masks[t]
                    q_ret_lst.append(q_ret.item())
                    q_ret = (rho_bar.gather(1,actions)[t]*(q_ret - q.gather(1,actions)[t])  + v[t])

                q_ret_lst.reverse()
                q_ret = torch.Tensor(q_ret_lst).reshape(-1, 1)

                adv = (q-v).data
                adv_ret = q_ret - v

                loss_trun = -(TIS.gather(1,actions)*adv_ret*logf.gather(1,actions)).mean()
                loss_bc = -(BC*f.data*adv*logf).sum(1, keepdim=True).mean() if not on_policy else 0.0
                loss_pi = actor_loss_weight*(loss_trun + loss_bc)
                entropy = beta*(f*logf).sum(1,keepdim=True).mean()
                loss_q = (q.gather(1,actions) - q_ret).pow(2).mean()/2.0

                if trust_region:

                    k = (-f_a/(f+eps)).data
                    g = (TIS*adv_ret)/(f+eps).data

                    if not on_policy:
                        g += (BC*f*adv/(eps+f)).data

                    kdotg = (k*g).sum(1, keepdim=True)
                    kdotk = (k*k).sum(1, keepdim=True)
                    scale = ((kdotg-max_kl)/kdotk).clamp(min=0).data

                    if (kdotk < eps**2).any():
                        scale[kdotk < eps**2] = 0.0

                    loss_trust = (actor_loss_weight*scale*(-f_a*(eps+f).log())).sum(1,keepdim=True).mean()
                    loss_pi += loss_trust

                optimizer.zero_grad()
                (loss_pi+loss_q+entropy).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 40.0)
                ensure_shared_grads(model, shared_model) # transfer gradients
                optimizer.step()
                update_params(average_policy, shared_model)



        ACER(on_policy=True)

        if memory.__len__() >= replay_start:
            for _ in range(np.random.poisson(replay_ratio)):
                ACER(on_policy=False)

    env.close()



def main():

    logs = mp.Array('d', range(10000))
    num_processes = mp.cpu_count()
    print('We got {} workers!!!'.format(num_processes))
    env = gym.make(env_name)
    S, A = env.observation_space.shape[0], env.action_space.n
    env.close()

    shared_model = ActorCritic(S, A).share_memory()
    average_policy = ActorCritic(S, A).share_memory()
    average_policy.load_state_dict(shared_model.state_dict())
    for param in average_policy.parameters():
        param.requires_grad = False

    if debug:
        train(average_policy, shared_model, 0, 1, S, A, logs)
    else:
        procs = []
        for i in range(num_processes):
            proc = mp.Process(target=train, args=(average_policy, shared_model, i, num_processes, S, A, logs))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()


if __name__ == "__main__":
    main()
