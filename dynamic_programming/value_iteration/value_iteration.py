import pdb
import numpy as np
import numpy.random as npr
import gym
import matplotlib.pyplot as plt

def value_iteration():
    # input: S, A, P, R, gamma
    env = gym.make("FrozenLake-v0")
    S, A = env.observation_space.n, env.action_space.n
    gamma = 1
    V = np.zeros(S)
    eps = 1e-8
    iter_num = 0

    # Value Iteration
    while True:
        delta = 0
        for s in range(S):
            V_old = V[s]
            Q = np.zeros(A)
            for a in range(A):
                for (prob, s_prime, rew, done) in env.P[s][a]:
                    Q[a] += prob*(rew + gamma*V[s_prime])
            V[s] = max(Q)
            delta = max(delta, abs(V[s]-V_old))
        iter_num += 1
        print("Iteration: {}\nValue: {}".format(iter_num, V))
        if delta<eps: break

    # Deterministic Policy from 'Optimal' Value Function
    pi = np.zeros(S)
    for s in range(S):
        Q = np.zeros(A)
        for a in range(A):
            for (prob, s_prime, rew, done) in env.P[s][a]:
                Q[a] += prob*(rew + gamma*V[s_prime])
        pi[s] = Q.argmax()

    # plotting
    vv = V.reshape(4,4)
    state = [['S','F','F','F'],['F','H','F','H'],['F','F','F','H'],['H','F','F','G']]
    plt.imshow(vv)
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(vv[i][j])[:5]+'/'+state[i][j], ha="center", va="center", color="brown")
    plt.title("Value Iteration for FrozenLake-v0 after {} Iterations".format(iter_num))
    plt.savefig('value_iteration.png')
    print(pi)
    return V, pi

if __name__ == "__main__":
    value_iteration()
