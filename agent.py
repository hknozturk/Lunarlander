import numpy as np
import random
from collections import namedtuple, deque

from memory import ReplayMemory, PrioritizedReplayMemory
from model import QNet, DuelingQNet

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, args, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.per = args.per
        self.dueling = args.dueling
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.lr = args.learning_rate
        self.update_freq = args.update_every
        # Q-Network
        if self.dueling:
            self.local_qnet = DuelingQNet(state_size, action_size, seed).to(device)
            self.target_qnet = DuelingQNet(state_size, action_size, seed).to(device)
        else:
            self.local_qnet = QNet(state_size, action_size, seed).to(device)
            self.target_qnet = QNet(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.local_qnet.parameters(), lr=self.lr)

        # Replay Memory
        if self.per:
            self.memory = PrioritizedReplayMemory(args, self.buffer_size)
        else:
            self.memory = ReplayMemory(action_size, self.buffer_size, self.batch_size, seed)
        self.t_step = 0 # init time step for updating every UPDATE_EVERY steps

    def step(self, state, action, reward, next_state, done):
        if self.per:
            self.memory.append(state, action, reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done) # save experience to replay memory.
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                if self.dueling:
                    self.learn_DDQN(self.gamma)
                else:
                    self.learn(self.gamma)
    
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_qnet.eval()
        with torch.no_grad():
            action_values = self.local_qnet(state)
        self.local_qnet.train()
    
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, gamma):
        if self.per:
            idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample()
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.target_qnet(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.local_qnet(states).gather(1, actions)

        # Compute loss - element-wise mean squared error
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss
        self.optimizer.zero_grad()
        if self.per:
            (weights * loss).mean().backward() # Backpropagate importance-weighted minibatch loss
        else:
            loss.backward()
        self.optimizer.step()

        if self.per:
            errors = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
            self.memory.update_priorities(idxs, errors)
        # Update target network
        self.soft_update(self.local_qnet, self.target_qnet, self.tau)

    def learn_DDQN(self, gamma):
        if self.per:
            idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample()
        # Get index of maximum value for next state from Q_expected
        Q_argmax = self.local_qnet(next_states).detach()
        _, a_prime = Q_argmax.max(1)
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.target_qnet(next_states).detach().gather(1, a_prime.unsqueeze(1))
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.local_qnet(states).gather(1, actions)

        # Compute loss
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss
        self.optimizer.zero_grad()
        if self.per:
            (weights * loss).mean().backward() # Backpropagate importance-weighted minibatch loss
        else:
            loss.backward()
        self.optimizer.step()
        
        if self.per:
            errors = np.abs((Q_expected - Q_targets).detach().cpu().numpy())
            self.memory.update_priorities(idxs, errors)
        # Update target network
        self.soft_update(self.local_qnet, self.target_qnet, self.tau)

    def soft_update(self, local_model, target_model, tau):
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
