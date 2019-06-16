import pdb
import numpy as np
import numpy.random as npr
import gym
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

eps = 0.1
num_episode = 1000000
print_freq = 10000
lr = 0.1

env = gym.make("FrozenLake-v0")
S, A = env.observation_space.n, env.action_space.n
gamma = 1

def e_greedy(policy, e=eps):
    if npr.rand() < e:
        return npr.randint(A)
    else:
        return policy

def plot(Q):
    vv = np.max(Q,1).reshape(4,4)
    state = [['S','F','F','F'],['F','H','F','H'],['F','F','F','H'],['H','F','F','G']]
    plt.imshow(vv)
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(vv[i][j])[:5]+'/'+state[i][j], ha="center", va="center", color="brown")
    plt.title("Q Learning for FrozenLake-v0 after {} Episodes".format(num_episode))
    plt.savefig('q_learning.png')

def q_learning():
    Q = np.zeros((S,A))
    pi = npr.randint(A, size=S)
    policy_unstable = True


    avg_return = 0.0
    for epi in range(num_episode):
        epi_return = 0.0
        s = env.reset()
        a = e_greedy(pi[s])
        while True:
            s_prime, r, done, _ = env.step(a)
            epi_return += r
            Q[s,a] = Q[s,a] + lr*(r + gamma*np.max(Q[s_prime]) - Q[s,a])
            pi[s] = Q[s].argmax()
            if done:
                avg_return += epi_return
                if (epi+1)%print_freq == 0:
                    print("Episode: {}    Return: {}".format(epi+1, avg_return/print_freq))
                    print(pi)
                    print('@'*60)
                    print(np.max(Q,1).reshape(4,4))
                    print('@'*60)
                    avg_return = 0.0
                break
            s, a = s_prime, e_greedy(pi[s_prime])
    print(epi)
    plot(Q)
    return Q, pi

if __name__ == "__main__":
    Q, pi = q_learning()
