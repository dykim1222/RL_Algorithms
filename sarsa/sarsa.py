import pdb
import numpy as np
import numpy.random as npr
import gym
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

eps = 0.1
num_episode = 1000002
print_freq = 100000
lr = 0.1

env = gym.make("FrozenLake-v0")
S, A = env.observation_space.n, env.action_space.n
gamma = 1
lamb = 0.9

def e_greedy(policy, e=eps):
    if npr.rand() < e:
        return npr.randint(A)
    else:
        return policy

def plot(Q, mode='forward'):
    vv = np.max(Q,1).reshape(4,4)
    state = [['S','F','F','F'],['F','H','F','H'],['F','F','F','H'],['H','F','F','G']]
    plt.imshow(vv)
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(vv[i][j])[:5]+'/'+state[i][j], ha="center", va="center", color="brown")
    if mode=='forward':
        plt.title("SARSA for FrozenLake-v0 after {} Episodes".format(num_episode))
        plt.savefig('sarsa.png')
    elif mode=='backward':
        plt.title("SARSA with Eligibility Trace({0:0.2f}) after {} Episodes".format(lamb, num_episode))
        plt.savefig('sarsa_eligibility_trace.png')


def sarsa():
    Q = np.zeros((S,A))
    pi = npr.randint(A, size=S)
    avg_return = 0.0
    
    for epi in range(num_episode):
        epi_return = 0.0
        s = env.reset()
        a = e_greedy(pi[s])
        
        while True:
            s_prime, r, done, _ = env.step(a)
            epi_return += r
            a_prime = e_greedy(pi[s_prime])
            Q[s,a] = Q[s,a] + lr*(r + gamma*Q[s_prime,a_prime] - Q[s,a])
            pi[s] = Q[s].argmax()
            
            if done:
                avg_return += epi_return
                if epi%print_freq == 0:
                    print("Episode: {}    Return: {}".format(epi, avg_return/print_freq))
                    print(pi)
                    print('@'*60)
                    print(np.max(Q,1).reshape(4,4))
                    print('@'*60)
                    avg_return = 0.0
                break
                
            s, a = s_prime, a_prime
            
    plot(Q)
    return Q, pi

def sarsa_eligibility_trace():
    
    Q = np.zeros((S,A))
    E = np.zeros((S,A))
    pi = npr.randint(A, size=S)
    lamb = 0.5
    avg_return = 0.0
    
    for epi in range(num_episode):
        epi_return = 0.0
        s = env.reset()
        a = e_greedy(pi[s])
        
        while True:
            s_prime, r, done, _ = env.step(a)
            epi_return += r
            a_prime = e_greedy(pi[s_prime])
            td_error = r + gamma*Q[s_prime,a_prime] - Q[s,a]
            E[s,a] += 1
            Q += lr*td_error*E
            E = gamma*lamb*E
            pi = Q.argmax(1)
            
            if done:
                avg_return += epi_return
                if epi%print_freq == 0:
                    print("Episode: {}    Return: {}".format(epi, avg_return/print_freq))
                    print(pi)
                    print('@'*60)
                    print(np.max(Q,1).reshape(4,4))
                    print('@'*60)
                    avg_return = 0.0
                break
                
            s, a = s_prime, a_prime
            
    plot(Q, 'backward')
    return Q, pi

if __name__ == "__main__":
    sarsa()
    sarsa_eligibility_trace()
