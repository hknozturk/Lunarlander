import numpy as np
import random
from collections import namedtuple, deque

from memory import ReplayMemory
from model import QNet, DuelingQNet

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, seed, dueling, buffer_size, batch_size, gamma, tau, lr, update_freq):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.dueling = dueling
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_freq = update_freq

        # Q-Network
        if dueling:
            self.local_qnet = DuelingQNet(state_size, action_size, seed).to(device)
            self.target_qnet = DuelingQNet(state_size, action_size, seed).to(device)
        else:
            self.local_qnet = QNet(state_size, action_size, seed).to(device)
            self.target_qnet = QNet(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.local_qnet.parameters(), lr=self.lr)

        # Replay Memory
        self.memory = ReplayMemory(action_size, self.buffer_size, self.batch_size, seed)
        self.t_step = 0 # init time step for updating every UPDATE_EVERY steps

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done) # save experience to replay memory.
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                if self.dueling:
                    self.learn_DDQN(experiences, self.gamma)
                else:
                    self.learn(experiences, self.gamma)
    
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
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
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
        loss.backward()
        self.optimizer.step()
        # Update target network
        self.soft_update(self.local_qnet, self.target_qnet, self.tau)

    def learn_DDQN(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
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
        loss.backward()
        self.optimizer.step()
        # Update target network
        self.soft_update(self.local_qnet, self.target_qnet, self.tau)

    def soft_update(self, local_model, target_model, tau):
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
