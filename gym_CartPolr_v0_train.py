import tensorflow as tf
import numpy as np
import gym
from DQN1 import DQN

env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

DQN_agent = DQN(N_ACTIONS=N_ACTIONS, N_STATES=N_STATES)
DQN_agent.creat_net()

for i_episode in range(400):
    s = env.reset() # the shape of s is (4,) , a simple list type

    ep_r = 0
    while True:
        env.render()
        a = DQN_agent.choose_action(s) # s is list

        # take action
        # the shape of s_ is (4,1)
        s_, r, done, info = env.step(a)
        

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        DQN_agent.store_transition(s, a, r, s_)

        ep_r += r
        if DQN_agent.MEMORY_COUNTER > DQN_agent.MEMORY_CAPACITY: # after the momery is filled
             DQN_agent.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
