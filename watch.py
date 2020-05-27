import gym
import torch

from agent import Agent

class WatchTrainedAgent:
    def __init__(self, env_name, dueling, eps, t, buffer_size, batch_size, gamma, lr, update_fr, tau):
        self.env_name = env_name
        self.dueling = dueling
        self.eps = eps
        self.t = t
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_fr = update_fr

        self.env = gym.make(env_name)
        self.env.seed(0)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.agent = Agent(
            state_size = self.n_states,
            action_size = self.n_actions,
            seed = 0,
            dueling = dueling,
            buffer_size = buffer_size,
            batch_size = batch_size,
            gamma = gamma,
            tau = tau,
            lr = lr,
            update_freq = update_fr
        )
    
    def watch(self):
        self.agent.local_qnet.load_state_dict(
            torch.load('saves/save-{}-{}_dueling-{}_eps-{}_t-{}_buffer-{}_batch-{}_gamma-{}_lr-{}_update_fr'.format(self.env_name, self.dueling, self.eps, self.t, self.buffer_size, self.batch_size, self.gamma, self.lr, self.update_fr), map_location = lambda storage, loc: storage))
            
        for i in range(3):
            state = self.env.reset()
            for j in range(2000):
                action = self.agent.act(state)
                self.env.render()
                state, reward, done, _ = self.env.step(action)

                if done:
                    state = self.env.reset()

        self.env.close()