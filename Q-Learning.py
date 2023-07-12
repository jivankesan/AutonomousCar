import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import NeuralNetwork


class MemoryReplay(object):
    # one time step is not sufficient for the model to learn long term correlations
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    #function to add desired memory to the array
    def push(self, state, action, reward, next_state, done):  # event includes, last result, current state, last action, last reward
        item = (state, action, reward, next_state, done)
        self.memory.append(item)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):  # function to convert a random sample of training data into tensors
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(states),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.tensor(next_states),
            torch.tensor(dones)
        )


class Dqn():
    def __init__(self, input_size=3, action=1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, action)
        self.memory = MemoryReplay(100000)
        self.optimizer = optim.Adam((self.model.parameters()), lr = 0.01) #can experiment with multiple optimizers
        self.prev_state = nn.Tensor(input_size).unsqueeze(0)
        self.prev_action = 0 #0 is for straight, (0,-1) is left, (1,0) is right
        self.prev_reward = 0

    def select_action(self, state):
        probs = f.softmax(self.model(Variable(state, volatile=True))*20) #temperature parameter = 7
        action = probs.multinomial()
        return action.data[0, 0]

    def learn(self, batch_state, batch_next, batch_reward, batch_action):




# add optimizer for SGD (using Adam)
