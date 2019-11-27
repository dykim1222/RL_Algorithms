import pdb
import numpy as np
import numpy.random as npr
import gym
import matplotlib.pyplot as plt

eps = 0.1
env = gym.make("FrozenLake-v0")
num_episode = 200000
S, A = env.observation_space.n, env.action_space.n
gamma = 1
print_freq = 10000

def e_greedy(policy, e=eps):
    if npr.rand() < e:
        return npr.randint(A)
    else:
        return policy

def monte_carlo_control():
    Q = np.zeros((S,A))
    N = np.zeros((S,A))
    pi = npr.randint(A, size=S)

    avg_return = 0.0
    for epi in range(num_episode):
        s_lst, a_lst, r_lst = [], [], []
        s = env.reset()
        a = e_greedy(pi[s])
        while True:
            s_lst.append(s)
            a_lst.append(a)
            s, r, done, _ = env.step(a)
            r_lst.append(r)
            a = e_greedy(pi[s])
            if done:
                avg_return += r_lst[-1]
                if epi%print_freq == 0:
                    avg_return = 0.0
                break

        for i in reversed(range(len(r_lst))):
            r_lst[i] = r_lst[i] + gamma*(r_lst[i+1] if i<len(r_lst)-1 else 0)
            s, a = s_lst[i], a_lst[i]
            s_a_pair = np.transpose(np.stack((s_lst[:i], a_lst[:i])))
            if np.sum([((s,a)==t.squeeze()).all(-1) for t in s_a_pair]) == 0:
                N[s,a] += 1
                Q[s,a] += (r_lst[i]-Q[s,a])/N[s,a]
                pi[s] = Q[s].argmax()

    vv = np.max(Q,1).reshape(4,4)
    state = [['S','F','F','F'],['F','H','F','H'],['F','F','F','H'],['H','F','F','G']]
    plt.imshow(vv)
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(vv[i][j])[:5]+'/'+state[i][j], ha="center", va="center", color="brown")
    plt.title("Monte Carlo Control for FrozenLake-v0 after {} Episodes".format(num_episode))
    plt.savefig('monte_carlo_control.png')
    return Q, pi

if __name__ == "__main__":
    monte_carlo_control()
