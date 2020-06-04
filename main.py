import argparse
import gym
import random
import torch
import numpy as np
from collections import deque
from plot_graph import PltResults
from watch import WatchTrainedAgent

parser = argparse.ArgumentParser(description="DQN")
parser.add_argument("--env", default="LunarLander-v2", help="GYM environment")
parser.add_argument("--per", type=bool, default=False, help="Use Prioritized Experience Replay")
parser.add_argument("--dueling", type=bool, default=True, help="Dueling DQN model or not")
parser.add_argument("--episodes", type=int, default=2000, help="Episodes")
parser.add_argument("--max_timestep", type=int, default=1000, help="Time steps (iterations) in an episode")
parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Replay buffer size")
parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--tau", type=float, default=1e-3, help="For soft update of target parameters")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon")
parser.add_argument("--eps_end", type=float, default=0.01, help="Minimum epsilon")
parser.add_argument("--eps_decay", type=float, default=0.995, help="Decay epsilon")
parser.add_argument("--update_every", type=int, default=2, help="How often to update the network")
parser.add_argument("--priority_exponent", type=float, default=0.6, metavar="ω", help="Prioritised experience replay exponent (originally denoted α)")
parser.add_argument("--priority_weight", type=float, default=0.4, metavar="β", help="Initial prioritised experience replay importance sampling weight")
args = parser.parse_args()

env = gym.make(args.env)
env.seed(0)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

from agent import Agent

agent = Agent(
    args,
    state_size = n_states,
    action_size = n_actions,
    seed = 0
)

def dqn(episodes=args.episodes, max_t=args.max_timestep, eps_start=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay):
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
            torch.save(agent.local_qnet.state_dict(), './results/model-{}-{}_per-{}_dueling-{}_eps-{}_t-{}_buffer-{}_batch-{}_gamma-{}_lr-{}_update_fr'.format(args.env, args.per, args.dueling, args.episodes, args.max_timestep, args.buffer_size, args.batch_size, args.gamma, args.learning_rate, args.update_every))

            np.save('./results/score-{}-{}_per-{}_dueling-{}_eps-{}_t-{}_buffer-{}_batch-{}_gamma-{}_lr-{}_update_fr'.format(args.env, args.per, args.dueling, args.episodes, args.max_timestep, args.buffer_size, args.batch_size, args.gamma, args.learning_rate, args.update_every), scores)
            break
    return scores


scores = dqn()

# Plot training scores of the agent
fig = PltResults()
fig.plot(scores)

# Watch trained agent
trainedAgent = WatchTrainedAgent(args)
trainedAgent.watch()