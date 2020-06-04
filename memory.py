import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    """
    Fixed size buffer to store experience tuples.
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        exp = Experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        TODO: instead of randomly sample a subset of experiences. Implement a prioritized replay
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        # Return the current size of internal memory.
        return len(self.memory)


from segment_tree import SegmentTree

class PrioritizedReplayMemory:
    def __init__(self, args, capacity):
        self.capacity = capacity
        self.discount = args.gamma
        self.priority_weight = args.priority_weight
        self.priority_exponent = args.priority_exponent
        self.absolute_error_upper = args.absolute_error_upper
        self.t = 0 # Internal episode timestep counter
        self.tree = SegmentTree(capacity) # Store experiences in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.priority_weight_increase = (1 - args.priority_weight) / self.capacity

    # Adds state and action at time t, reward and done at time t + 1
    def append(self, state, action, reward, next_state, done):
        self.tree.append(Experience(state, action, reward, next_state, done), self.tree.max) # Store new transition with maximum priority
        self.t = 0 if done else self.t + 1 # Start new episodes with t = 0

    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment) # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.tree.find(sample) # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probability 0
            if prob != 0:
                valid = True # Note that conditions are valid but extra conservative around buffer index 0

        experience = self.tree.get(idx)

        return prob, idx, tree_idx, experience

    def sample(self, batch_size):
        self.priority_weight = min(self.priority_weight + self.priority_weight_increase, 1)
        p_total = self.tree.total() # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size # Batch size number of segments, based on sum over all probabilities

        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)] # Get batch of valid samples
        probs, idxs, tree_idxs, experiences = zip(*batch)
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.long)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None])).to(device=device).to(dtype=torch.float32)

        probs = np.array(probs, dtype=np.float32) / p_total # Calculate normalised probabilities
        capacity = self.capacity if self.tree.full else self.tree.index
        weights = (capacity * probs) ** -self.priority_weight # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=device) # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, rewards, next_states, dones, weights

    def update_priorities(self, idxs, priorities):
        # priorities = errors
        clipped_errors = np.minimum(priorities, self.absolute_error_upper)
        clipped_errors = np.power(clipped_errors, self.priority_exponent)
        for idx, priority in zip(idxs, clipped_errors):
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)