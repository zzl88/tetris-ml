import collections
import logging
import numpy as np
import random
import torch
import torch.nn as nn
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
        self._layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=2, stride=1),
                                     nn.ReLU())
        self._layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU())
        self._layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1), nn.ReLU())
        self._layer4 = nn.Sequential(nn.Linear(384, 512), nn.ReLU())
        self._layer5 = nn.Linear(512, self._nodes[-1])

    def forward(self, x):
        out = self._layer1(x)
        out = self._layer2(out)
        out = self._layer3(out)
        # print(out.size())
        out = out.view(out.size()[0], -1)
        out = self._layer4(out)
        out = self._layer5(out)
        return out

    def remember(self, old_state, action, reward, new_state, done):
        # reward = torch.from_numpy(np.array([reward],
        #                                    dtype=np.float32)).unsqueeze(0)
        self._memory.append((old_state, action, reward, new_state, done))

    def predict(self, old_state):
        with torch.no_grad():
            old_state = torch.from_numpy(old_state).float()[None, None, ...]
            prediction = self(old_state)
            d = prediction.detach().cpu().numpy()[0]
            # LOGGER.info(d)
            return np.argmax(d)

    def reset_memory(self):
        self._memory = []

    def replay_memory(self):
        LOGGER.info(f'replaying size[{len(self._memory)}]')
        for param in super().parameters():
            LOGGER.debug(param)
        for old_state, action, reward, next_state, done in self._memory:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.from_numpy(next_state).float()[None,
                                                                     None, ...]
            state_tensor = torch.from_numpy(old_state).float()[None, None, ...]
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
