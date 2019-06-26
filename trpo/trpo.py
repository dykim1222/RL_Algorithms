import math
import gym
import random
import numpy as np
import numpy.random as npr
from collections import deque, namedtuple
# !pip install tensorboardX
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import scipy.optimize
import matplotlib.pyplot as plt

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


buffer_size = 100000
num_episodes = 1000
env_name = "MountainCarContinuous-v0"
render = False

gamma = 0.995
lam = 0.97
l2_reg = 1e-3
max_kl = 1e-2
cg_iters = 10
cg_damping = 1e-1
alpha = 0.5

seed = 10
print_freq = 10


class Actor(nn.Module):
    def __init__(self, num_obs, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_obs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        self.log_std = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, obs):
        obs = obs
        obs = torch.tanh(self.fc1(obs))
        obs = torch.tanh(self.fc2(obs))
        mu = self.fc3(obs)
        log_std = self.log_std.expand_as(mu)
        std = torch.exp(log_std)
        return mu, log_std, std

    def select_action(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0)
        mu, log_std, std = self.forward(obs)
        return torch.normal(mu, std)

    def set_params_to(self, theta):
        idx = 0
        for param in self.parameters():
            length = int(np.prod(list(param.shape)))
            param.data.copy_(theta[idx:idx + length].view(param.size()))
            idx += length

    def get_kl(self, obss):
        mu, log_std, std = self.forward(obss)
        mu0 = mu.data
        log_std0 = log_std.data
        std0 = std.data
        kl = log_std - log_std0 + (std0.pow(2) + (mu0 - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_logprob(self, obss, acts, requires_grad=True):
        with torch.autograd.set_grad_enabled(requires_grad):
            mu, log_std, std = self.forward(obss)
        var = std.pow(2)
        log_density = -(acts - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, num_obs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_obs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, obs):
        obs = obs
        obs = torch.tanh(self.fc1(obs))
        obs = torch.tanh(self.fc2(obs))
        obs = self.fc3(obs)
        return obs

    def set_params_to(self, theta):
        idx = 0
        for param in self.parameters():
            length = int(np.prod(list(param.shape)))
            param.data.copy_(theta[idx:idx + length].view(param.size()))
            idx += length

Transition = namedtuple('Transition',('state', 'action', 'mask', 'next_state','reward'))

class Memory(object):
    def __init__(self):
        self.memory = deque(maxlen=buffer_size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

    def get_batch(self):
        batch = self.sample()
        rews = torch.Tensor(batch.reward)
        mask = torch.Tensor(batch.mask)
        acts = torch.Tensor(np.concatenate(batch.action, 0))
        obss = torch.Tensor(batch.state)
        return obss, acts, rews, mask

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))
    flat_grad = torch.cat(grads)
    return flat_grad

def flatten(item):
    return torch.cat([i.view(-1) for i in item]).contiguous()

def estimate_advantage(values, rews, mask):
    rets = torch.Tensor(rews.size(0),1)
    dels = torch.Tensor(rews.size(0),1)
    advs = torch.Tensor(rews.size(0),1)
    prev_ret = 0
    prev_val = 0
    prev_adv = 0
    for i in reversed(range(rews.size(0))):
        rets[i] = rews[i]  + gamma * prev_ret * mask[i]
        dels[i] = rews[i]  + gamma * prev_val * mask[i] - values.data[i]
        advs[i] = dels[i] + gamma * lam * prev_adv * mask[i]
        prev_ret = rets[i, 0]
        prev_val = values.data[i, 0]
        prev_adv = advs[i, 0]
    advs = (advs - advs.mean()) / advs.std()
    return rets, advs

def linesearch(model, f, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    theta0 = flatten(model.parameters())
    fval = f(False).data
    for j in range(max_backtracks):
        theta_new = theta0 + (alpha**j)* fullstep
        model.set_params_to(theta_new)
        newfval = f(False).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * alpha**j
        ratio = actual_improve / expected_improve
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return theta_new
    return theta0

def conj_grad(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def update_params(pnet, vnet, memory):
    obss, acts, rews, mask = memory.get_batch()
    with torch.no_grad(): values = vnet(obss)
    rets, advs = estimate_advantage(values, rews, mask)

    # optimize critic
    def get_value_loss(flat_params):
        vnet.set_params_to(torch.Tensor(flat_params))
        for param in vnet.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_ = vnet(obss)
        value_loss = (values_ - rets).pow(2).mean()
        # weight decay
        for param in vnet.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(vnet).data.double().numpy())
    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, flatten(vnet.parameters()).data.double().numpy(), maxiter=25)
    vnet.set_params_to(torch.Tensor(flat_params))

    def hess_vec(v):
        kl = pnet.get_kl(obss).mean()
        g = flatten(torch.autograd.grad(kl, pnet.parameters(), create_graph=True))
        gv = g.dot(v)
        Hv = flatten(torch.autograd.grad(gv, pnet.parameters())).data
        return Hv + cg_damping*v

    log_prob_old = pnet.get_logprob(obss, acts, False)

    def surr_loss(requires_grad=True):
        log_prob = pnet.get_logprob(obss, acts, requires_grad)
        loss = -advs * torch.exp(log_prob - log_prob_old)
        return loss.mean()

    # use conj grad algo to compute x=inv(H)*g
    loss = surr_loss()
    loss_grad = flatten(torch.autograd.grad(loss, pnet.parameters())).data
    stepdir = conj_grad(hess_vec, -loss_grad, nsteps=cg_iters)
    scale = torch.sqrt(2*max_kl/(stepdir.dot(hess_vec(stepdir))))
    # print(("lagrange multiplier:", 1/scale, "grad_norm:", loss_grad.norm()))

    # update the policy by backtracking line search
    fullstep = scale * stepdir
    first_order_approx = (-loss_grad * fullstep).sum(0, keepdim=True)
    theta = linesearch(pnet, surr_loss, fullstep, first_order_approx)
    pnet.set_params_to(theta)

    return loss_grad.norm()




def main():
    env = gym.make(env_name)
    num_obs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    writer = SummaryWriter()
    pnet = Actor(num_obs, num_actions)
    vnet  = Critic(num_obs)

    running_state = ZFilter((num_obs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)

    score_lst = []
    avg_score = 0
    for epi in range(num_episodes):
        memory = Memory()
        obs = env.reset()
        obs = running_state(obs)
        score = 0
        for _ in range(10000):
            if render: env.render()
            action = pnet.select_action(obs).data[0].numpy()
            obs_new, rew, done, _ = env.step(action)
            score += rew
            obs_new = running_state(obs_new)
            mask = 0 if done else 1
            memory.push(obs, np.array([action]), mask, obs_new, rew)
            if done:
                avg_score += score/print_freq
                writer.add_scalar('Return', score, epi)
                score_lst.append(score)
                break
            obs = obs_new
        grad = update_params(pnet, vnet, memory)
        writer.add_scalar('loss_grad_norm', grad.item(), epi)

        if (epi+1)%print_freq==0:
            writer.add_scalar('Avg Return over last {} runs'.format(print_freq), avg_score, epi)
            print('Episode: %3d \t Avg Return: %.3f'%(epi+1, avg_score))
            avg_score = 0

    env.close()
    writer.close()

    plt.plot(score_lst)
    plt.show()

if __name__ == "__main__":
    main()
