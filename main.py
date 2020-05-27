import gym
import random
import torch
import numpy as np
from collections import deque
from plot_graph import PltResults
from watch import WatchTrainedAgent

ENV = 'LunarLander-v2'      # GYM environment
EPISODES = 2000             # Episodes
MAX_T = 1000                # Time steps (Iterations)
BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # for soft update of target parameters (temperature)
LEARNING_RATE = 5e-4        # learning rate
UPDATE_EVERY = 2            # how often to update the network
DUELING = True              # Dueling DQN model or DQN model

env = gym.make(ENV)
env.seed(0)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

from agent import Agent

agent = Agent(
    state_size = n_states,
    action_size = n_actions,
    seed = 0,
    dueling = DUELING,
    buffer_size = BUFFER_SIZE,
    batch_size = BATCH_SIZE,
    gamma = GAMMA,
    tau = TAU,
    lr = LEARNING_RATE,
    update_freq = UPDATE_EVERY
)

def dqn(episodes=EPISODES, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for episode in range(1, episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)  # update QNetwork
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.local_qnet.state_dict(), 'saves/save-{}-{}_dueling-{}_eps-{}_t-{}_buffer-{}_batch-{}_gamma-{}_lr-{}_update_fr'.format(ENV, DUELING, EPISODES, MAX_T, BUFFER_SIZE, BATCH_SIZE, GAMMA, LEARNING_RATE, UPDATE_EVERY))
            break
    return scores


scores = dqn()

# Plot training scores of the agent
fig = PltResults()
fig.plot(scores)

# Watch trained agent
trainedAgent = WatchTrainedAgent(ENV, DUELING, EPISODES, MAX_T, BUFFER_SIZE, BATCH_SIZE, GAMMA, LEARNING_RATE, UPDATE_EVERY, TAU)
trainedAgent.watch()