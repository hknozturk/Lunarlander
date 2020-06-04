import gym
import torch

from agent import Agent

class WatchTrainedAgent:
    def __init__(self, args):
        self.env_name = args.env
        self.per = args.per
        self.dueling = args.dueling
        self.eps = args.episodes
        self.t = args.max_timestep
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.update_fr = args.update_every

        self.env = gym.make(args.env)
        self.env.seed(0)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.agent = Agent(
            args,
            state_size = self.n_states,
            action_size = self.n_actions,
            seed = 0
        )
    
    def watch(self):
        self.agent.local_qnet.load_state_dict(
            torch.load('results/model-{}-{}_per-{}_dueling-{}_eps-{}_t-{}_buffer-{}_batch-{}_gamma-{}_lr-{}_update_fr'.format(self.env_name, self.per, self.dueling, self.eps, self.t, self.buffer_size, self.batch_size, self.gamma, self.lr, self.update_fr), map_location = lambda storage, loc: storage))
            
        for i in range(3):
            state = self.env.reset()
            for j in range(2000):
                action = self.agent.act(state)
                self.env.render()
                state, reward, done, _ = self.env.step(action)

                if done:
                    state = self.env.reset()

        self.env.close()