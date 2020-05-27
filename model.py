import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_size = 64, fc2_size = 64):
        """
        network consists of three connected layers.
        fc1 is the input layer. Takes same size tensor as state size and outputs a tensor that is the size of fc1 hidden nodes.
        fc2 is the second hidden layer. It takes in the the output of fc1 and outputs a tensor that is the size of fc2 hidden nodes.
        output layer takes input as fc2 layer output and outputs a tensor that is the size of actions.
        """
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.output = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        """
        Softmax function as activation layer after the final output layer.
        Softmax function will normalize the number of each possible action so that they all add up to 1. By doing this we know the correct probability of taking each action.
        We could then calculate the cross entropy loss to find out how far off our predictions were.
        Instead we use the pytorch class nn.CrossEntropyLoss later on in the project. This class carries out both the Softmax activation and cross entropy loss in one in order to provide a more stable function.

        REMEMBER: We need to apply the softmax activation function when we want to see the probability of taking an action given our current state.
        """
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action = self.output(x)
        # action_probabilities = F.softmax(self.output(x), dim=1)
        return action

class DuelingQNet(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_size = 64, fc2_size = 64):
        super(DuelingQNet, self).__init__()
        self.n_actions = action_size
        fc3_1_size = fc3_2_size = 32
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # Here we seperate into two streams
        # The one that calculate V(s)
        self.fc3_1 = nn.Linear(fc2_size, fc3_1_size)
        self.fc4_1 = nn.Linear(fc3_1_size, 1)
        # The one that calculate A(s, a)
        self.fc3_2 = nn.Linear(fc2_size, fc3_2_size)
        self.fc4_2 = nn.Linear(fc3_2_size, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        val = F.relu(self.fc3_1(x))
        val = self.fc4_1(val)

        adv = F.relu(self.fc3_2(x))
        adv = self.fc4_2(adv)
        # Q(s, a) = V(s) + (A(s, a) - 1 / |A| * sumA(s, a'))
        action = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.n_actions)
        return action