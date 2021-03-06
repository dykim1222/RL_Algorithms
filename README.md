# Deep Reinforcement Learning Algorithms
![dd](https://dv-website.s3.amazonaws.com/uploads/2018/06/pg_fundDRL_062718.png)

* Implementing RL algorithms in PyTorch. 
* Everything is included in one file.
* Teseted on simple environments (e.g. `FrozenLake-v0`, `CartPole-v1`, `Pendulum-v1`, etc.)
* Each algorithm is implemented to work fully in the mentioned environment. 
* To test on other environments, a hyperparameter search might be necessary.

## Requirements
```
gym, torch
```

## Algorithms

More Coming!

### Basic
1. [Dynamic Programming](https://github.com/dykim1222/RL_Algorithms/tree/master/dynamic_programming) (Policy Iteration and Value Iteration)
1. [Monte Carlo Control](https://github.com/dykim1222/RL_Algorithms/tree/master/monte_carlo_control)
1. [On-Policy Control with SARSA](https://github.com/dykim1222/RL_Algorithms/tree/master/sarsa)
1. [Q Learning](https://github.com/dykim1222/RL_Algorithms/tree/master/q_learning)

### Policy Gradients

1. [REINFORCE](https://github.com/dykim1222/RL_Algorithms/tree/master/reinforce)
1. [Actor Critic](https://github.com/dykim1222/RL_Algorithms/tree/master/actor_critic)
1. [Deep Deterministic Policy Gradient(DDPG)](https://github.com/dykim1222/RL_Algorithms/tree/master/ddpg)
1. [Trust Region Policy Optimization with Generalized Advantage Estimation (TRPO/GAE)](https://github.com/dykim1222/RL_Algorithms/tree/master/trpo) ([ref](https://github.com/ikostrikov/pytorch-trpo))
1. [Asynchronous Advantage Actor Critic (A3C)](https://github.com/dykim1222/RL_Algorithms/tree/master/a3c)
1. [Sample Efficient Actor-Critic with Experience Replay (ACER)](https://github.com/dykim1222/RL_Algorithms/tree/master/acer)
1. [Proximal Policy Optimization (PPO)](https://github.com/dykim1222/RL_Algorithms/tree/master/ppo)

### Deep Q-Learning

1. [Deep Q-Network (DQN)](https://github.com/dykim1222/RL_Algorithms/tree/master/dqn)
1. [Double Deep Q-Network (DDQN)](https://github.com/dykim1222/RL_Algorithms/tree/master/ddqn)
1. [Dueling Deep Q-Network (Dueling DQN)](https://github.com/dykim1222/RL_Algorithms/tree/master/duelingdqn)

## Resources
* [David Silver's RL Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
* [Sutton and Barto's Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
* This project is inspired by [seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL).
