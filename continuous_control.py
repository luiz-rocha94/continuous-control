# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 22:13:40 2021

@author: rocha
"""

from unityagents import UnityEnvironment
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline

from ddpg_agent import Agent

env = UnityEnvironment(file_name=r'D:\deep-reinforcement-learning\p2_continuous-control\Reacher_Windows_x86_64\Reacher.exe')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# size of each action
action_size = brain.vector_action_space_size
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
# create the agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)

def ddpg(n_episodes=100):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        state = env_info.vector_observations[0]                # get the current state
        agent.reset()
        score = 0                                              # initialize the score
        while True:
            action = agent.act(state)                          # select an action
            env_info = env.step(action)[brain_name]            # send action to tne environment
            next_state = env_info.vector_observations[0]       # get next state
            reward = env_info.rewards[0]                       # get reward
            done = env_info.local_done[0]                      # see if episode finished
            score += reward                                    # update the score
            state = next_state                                 # roll over state to next time step
            if done:                                           # exit loop if episode finished
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if np.mean(scores_deque)>=30:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
state = env_info.vector_observations[0]                # get the current state
agent.reset()
score = 0                                              # initialize the score
while True:
    action = agent.act(state)                          # select an action
    env_info = env.step(action)[brain_name]            # send action to tne environment
    next_state = env_info.vector_observations[0]       # get next state
    reward = env_info.rewards[0]                       # get reward
    done = env_info.local_done[0]                      # see if episode finished
    score += reward                                    # update the score
    state = next_state                                 # roll over state to next time step
    if done:                                           # exit loop if episode finished
        break
        
env.close()