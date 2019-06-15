import pdb
import numpy as np
import numpy.random as npr
import gym
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def policy_iteration():
    # input: S, A, P, R, gamma
    env = gym.make("FrozenLake-v0")
    S, A = env.observation_space.n, env.action_space.n
    gamma = 1
    V = np.zeros(S)
    pi = npr.randint(A, size=S)
    policy_unstable = True
    eps = 1e-6
    iter_num = 0

    while policy_unstable:

        # policy evaluation
        delta = 1
        while delta > eps:
            delta = 0
            for s in range(len(V)):
                V_old = V[s]
                trans = env.P[s][pi[s]]
                value = 0
                for (prob, s_prime, rew, done) in trans:
                    value += prob*(rew + gamma*V[s_prime])
                V[s] = value
                delta = max(delta, abs(V_old-V[s]))

        # policy improvement
        policy_unstable = False
        for s in range(len(V)):
            pi_old = pi[s]
            Q = np.zeros(A)
            for a in range(A):
                trans = env.P[s][a]
                for (prob, s_prime, rew, done) in trans:
                    Q[a] += prob*(rew + gamma*V[s_prime])
            pi[s] = Q.argmax()
            if pi_old != pi[s]: policy_unstable=True

        iter_num += 1
        print("Iteration: {}    Policy: {}\nValue: {}".format(iter_num, pi, V))


    # plotting
    vv = V.reshape(4,4)
    state = [['S','F','F','F'],['F','H','F','H'],['F','F','F','H'],['H','F','F','G']]
    plt.imshow(vv)
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(vv[i][j])[:5]+'/'+state[i][j], ha="center", va="center", color="brown")
    plt.title("Policy Iteration for FrozenLake-v0 after {} Iterations".format(iter_num))
    plt.savefig('policy_iteration.png')
    return V, pi

if __name__ == "__main__":
    policy_iteration()
