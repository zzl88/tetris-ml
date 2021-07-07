import collections
import logging
import numpy as np
import random
import torch.nn
import torch.nn.functional as F
from typing import List

from torch import optim

_DEVICE = 'cpu'

LOGGER = logging.getLogger(__name__)


class Agent(torch.nn.Module):
    def __init__(self, gamma: float, layers: List[int], memory_size,
                 learning_rate):
        super(Agent, self).__init__()
        self._gamma = gamma
        self._memory = []
        self._nodes = layers
        self._network(layers)
        self._optimizer = optim.Adam(super().parameters(),
                                     weight_decay=0,
                                     lr=learning_rate)

    def load_weights(self, path):
        super().load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(super().state_dict(), path)

    def _network(self, layers):
        ll = []
        for i in range(len(layers)):
            if i > len(layers) - 1:
                break
            ll.append(torch.nn.Linear(layers[i], layers[i + 1]))
        self._layers = torch.nn.ModuleList(ll)

    def forward(self, x):
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        x = F.softmax(self._layers[-1](x), dim=-1)
        return x

    def remember(self, old_state, action, reward, new_state, done):
        self._memory.append((old_state, action, reward, new_state, done))

    def predict(self, old_state):
        with torch.no_grad():
            state_old_tensor = torch.tensor(old_state.reshape(
                (1, self._nodes[0])),
                                            dtype=torch.float32).to(_DEVICE)
            prediction = self(state_old_tensor)
            return np.argmax(prediction.detach().cpu().numpy()[0])

    def reset_memory(self):
        self._memory = []

    def replay_memory(self):
        LOGGER.info(f'replaying size[{len( self._memory)}]')
        for old_state, action, reward, next_state, done in self._memory:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(next_state.reshape(
                (1, self._nodes[0])),
                                             dtype=torch.float32).to(_DEVICE)
            state_tensor = torch.tensor(old_state.reshape((1, self._nodes[0])),
                                        dtype=torch.float32,
                                        requires_grad=True).to(_DEVICE)
            if not done:
                target = reward + self._gamma * torch.max(
                    self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][action] = target
            target_f.detach()
            self._optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self._optimizer.step()

    def train_short_memory(self, old_state, action, reward, new_state, done):
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(new_state.reshape(
            (1, self._nodes[0])),
                                         dtype=torch.float32).to(_DEVICE)
        state_tensor = torch.tensor(old_state.reshape((1, self._nodes[0])),
                                    dtype=torch.float32,
                                    requires_grad=True).to(_DEVICE)
        if not done:
            target = reward + self._gamma * torch.max(
                self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][action] = target
        target_f.detach()
        self._optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self._optimizer.step()
