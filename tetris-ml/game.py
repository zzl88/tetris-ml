import datetime
import logging

import numpy as np
import pandas as pd
import pygame.display
import random
import time
from enum import Enum

LOGGER = logging.getLogger(__name__)


class Tetrimino(Enum):
    I = 1
    O = 2
    T = 3
    J = 4
    L = 5
    S = 6
    Z = 7

    @classmethod
    def gen(cls):
        return Tetrimino(random.randint(1, 7))

    def points(self):
        """
        rect is in unit size (len = 1)
        return [ the center coordinate of a rect ]
        """
        if self == Tetrimino.I:
            return np.array([[-2, 0], [-1, 0], [0, 0], [1, 0]])
        if self == Tetrimino.O:
            return np.array([[-1, 0], [0, 0], [0, -1], [-1, -1]])
        if self == Tetrimino.T:
            return np.array([[-1, 0], [0, 0], [1, 0], [0, -1]])
        if self == Tetrimino.J:
            return np.array([[0, 1], [0, 0], [0, -1], [-1, -1]])
        if self == Tetrimino.L:
            return np.array([[-1, 1], [-1, 0], [-1, -1], [0, -1]])
        if self == Tetrimino.S:
            return np.array([[0, 0], [-1, 0], [-1, -1], [-2, -1]])
        if self == Tetrimino.Z:
            return np.array([[-1, 0], [0, 0], [0, -1], [1, -1]])


class Key(Enum):
    UP = 1  # 1 0 0 0
    RIGHT = 2  # 0 1 0 0
    DOWN = 3  # 0 0 1 0
    LEFT = 4  # 0 0 0 1


_RECT_LEN = 20


class Game(object):
    def __init__(self, speed=1):
        pygame.display.set_caption('Tetris')
        self._speed = speed
        self._width = 10
        self._height = 20
        if self._speed > 0:
            self._window = pygame.display.set_mode(
                (self._width * _RECT_LEN, self._height * _RECT_LEN))
        # 0 for empty, 1 for filled
        self._board = pd.DataFrame(np.array(
            [np.repeat(0, self._width) for i in range(self._height)]),
                                   columns=[i for i in range(self._width)])
        # print(self._board)
        self._tick = 0
        self._score = 0
        self._gen_next()

    def score(self):
        return self._score

    def run(self):
        pygame.init()
        clock = pygame.time.Clock()
        run = True
        last = time.time()
        while run:
            clock.tick(120)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self._handle_move(Key.DOWN)
                    elif event.key == pygame.K_UP:
                        self._handle_move(Key.UP)
                    elif event.key == pygame.K_RIGHT:
                        self._handle_move(Key.RIGHT)
                    elif event.key == pygame.K_LEFT:
                        self._handle_move(Key.LEFT)
            now = time.time()
            if self._speed == 0 or now - last >= 1 / self._speed:
                # print('dropping 1')
                # print(self._cur_pos)
                self._tick += 1
                if not self._move_down():
                    if not self._freeze():
                        run = False
                last = now
            self._update_ui()

        pygame.quit()

    def train(self, agent, batch_size, epsilon=0.2):
        if self._speed > 0:
            pygame.init()
        run = True
        cur_state = self._get_state()
        # LOGGER.info(cur_state)
        agent.reset_memory()

        reward_acc = 0

        def _train(action, run, reward):
            nonlocal cur_state, reward_acc
            state = self._get_state()
            # agent.train_short_memory(cur_state, action, reward, state, run)
            agent.remember(cur_state, action, reward, state, run)
            cur_state = state
            reward_acc += reward

        def _wait_ui():
            if self._speed > 0:
                pygame.time.wait(1000 // self._speed // 1)
                self._update_ui()

        while run:
            if self._speed > 0:
                quit = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit = True
                if quit:
                    break

            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 39)
                LOGGER.debug(f'random action[{action}]')
            else:
                action = agent.predict(cur_state)
                LOGGER.debug(f'predict action[{action}]')

            rotate = action // 10
            for i in range(rotate):
                self._rotate()
                _wait_ui()

            move = int(action % 10 - self._width / 2)
            for i in range(abs(move)):
                self._move_h(1 if move > 0 else -1)
                _wait_ui()

            while self._move_down():
                _wait_ui()

            score = self._score
            if not self._freeze():
                run = False
            _train(action, run, self._score - score + 4 if run else -100)

            _wait_ui()
        if self._speed > 0:
            pygame.quit()
        agent.replay_memory()
        # LOGGER.info(f'{self._board}')
        return reward_acc

    def _handle_move(self, key):
        # print(key)
        if key == Key.UP:
            return self._rotate()
        elif key == Key.DOWN:
            return self._move_down()
        elif key == Key.LEFT:
            return self._move_h(-1)
        elif key == Key.RIGHT:
            return self._move_h(1)
        return None

    def _min_x(self, cur=None):
        if cur is None:
            cur = self._cur
        return np.amin(cur, axis=0)[0]

    def _max_x(self, cur=None):
        if cur is None:
            cur = self._cur
        return np.amax(cur, axis=0)[0]

    def _min_y(self, cur=None):
        if cur is None:
            cur = self._cur
        return np.amin(cur, axis=0)[1]

    def _max_y(self, cur=None):
        if cur is None:
            cur = self._cur
        return np.amax(cur, axis=0)[1]

    def _rotate(self) -> True:
        # we rotate 90 degrees at a time, just play the trick
        cur = np.array([(y, -x) for x, y in self._cur])
        if self._will_collide(cur, self._cur_pos):
            return False

        min_y = self._min_y(cur)
        if self._cur_pos[1] + min_y < 0:
            return False
        min_x = self._min_x(cur)
        if self._cur_pos[0] + min_x < 0:
            if not self._move_h(-min_x - self._cur_pos[0]):
                return False
        max_x = self._max_x(cur)
        if self._cur_pos[0] + max_x >= self._width:
            if not self._move_h(-max_x - self._cur_pos[0] + self._width - 1):
                return False
        max_y = self._max_y(cur)
        if self._cur_pos[1] + max_y >= self._height:
            if self._will_collide(
                    cur, [self._cur_pos[0], self._height - 1 - max_y]):
                return False
            self._cur_pos[1] = self._height - 1 - max_y
        self._cur = cur
        # print(f'{self._cur_pos}')
        # print(self._cur)
        return True

    def _move_h(self, by) -> bool:
        if by == 0:
            return True

        if by < 0:
            if self._cur_pos[0] + self._min_x(
            ) + by >= 0 and not self._will_collide(
                    self._cur, [self._cur_pos[0] + by, self._cur_pos[1]]):
                self._cur_pos[0] += by
                return True
        elif by > 0:
            if self._cur_pos[0] + self._max_x(
            ) + by < self._width and not self._will_collide(
                    self._cur, [self._cur_pos[0] + by, self._cur_pos[1]]):
                self._cur_pos[0] += by
                return True
        return False

    def _move_down(self) -> bool:
        if self._cur_pos[1] + self._min_y() > 1 and not self._will_collide(
                self._cur, [self._cur_pos[0], self._cur_pos[1] - 1]):
            self._cur_pos[1] -= 1
            return True
        return False

    def _freeze(self):
        for x, y in self._cur:
            self._board.at[int(self._cur_pos[1] + y),
                           int(self._cur_pos[0] + x)] = 1
        df = self._board
        df['sum'] = df.sum(axis=1)
        temp = df[df['sum'] == self._width]
        if not temp.empty:
            self._score += temp['sum'].sum()
            df.drop(index=temp.index, inplace=True)
            for i in range(len(temp.index)):
                df.loc[-1 * i - 1] = 0
            df.index = np.arange(0, self._height, 1)
            LOGGER.info(f'score[{self._score}]')
        self._board.drop(columns=['sum'], inplace=True)
        self._gen_next()
        return not self._will_collide(self._cur, self._cur_pos)

    def _will_collide(self, points, cur_pos):
        for x, y in points:
            y = int(cur_pos[1] + y)
            x = int(cur_pos[0] + x)
            if y < 0 or y >= self._height or x < 0 or x >= self._width:
                continue
            if self._board.at[y, x] == 1:
                return True
        return False

    def _update_ui(self):
        self._window.fill(0)

        for y in range(self._height):
            for x in range(self._width):
                if self._board.at[y, x] == 1:
                    pygame.draw.rect(self._window, (255, 255, 255),
                                     self._rect(x, y))
        for r in self._cur:
            pygame.draw.rect(
                self._window, (255, 255, 255),
                self._rect(r[0] + self._cur_pos[0], r[1] + self._cur_pos[1]))
        pygame.display.flip()

    def _rect(self, x, y):
        rect = pygame.Rect(x * _RECT_LEN, (self._height - y) * _RECT_LEN,
                           _RECT_LEN, _RECT_LEN)
        return rect

    def _gen_next(self):
        # center of the current tetrimino
        self._cur_pos = [self._width / 2, self._height]
        self._cur = Tetrimino.gen().points()
        if self._max_y() + self._cur_pos[1] >= self._height:
            self._cur_pos[1] = self._height - 1 - self._max_y()
        # print(f'gen_next')
        # print(f'{self._cur_pos}')
        # print(f'{self._cur}')

    def _get_state(self):
        df = self._board.copy()
        for r in self._cur:
            df.at[r[1] + self._cur_pos[1], r[0] + self._cur_pos[0]] = 1
        LOGGER.debug(self._cur_pos)
        LOGGER.debug(self._cur)
        LOGGER.debug(df.shape)
        return df.to_numpy()
