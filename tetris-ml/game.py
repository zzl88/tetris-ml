import numpy as np
import pandas as pd
import pygame.display
import random
import time
from enum import Enum


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
            return np.array([[-1.5, 0.5], [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5]])
        if self == Tetrimino.O:
            return np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5],
                             [-0.5, -0.5]])
        if self == Tetrimino.T:
            return np.array([[-0.5, 0.5], [0.5, 0.5], [1.5, 0.5], [0.5, -0.5]])
        if self == Tetrimino.J:
            return np.array([[0.5, 1.5], [0.5, 0.5], [0.5, -0.5], [-0.5,
                                                                   -0.5]])
        if self == Tetrimino.L:
            return np.array([[-0.5, 1.5], [-0.5, 0.5], [-0.5, -0.5],
                             [0.5, -0.5]])
        if self == Tetrimino.S:
            return np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5],
                             [-1.5, -0.5]])
        if self == Tetrimino.Z:
            return np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [1.5,
                                                                    -0.5]])


_RECT_LEN = 20


class Game(object):
    def __init__(self, speed=1):
        pygame.display.set_caption('Tetris')
        self._speed = speed
        self._width = 10
        self._height = 20
        self._window = pygame.display.set_mode(
            (self._width * _RECT_LEN, self._height * _RECT_LEN))
        # 0 for empty, 1 for filled
        self._board = pd.DataFrame(np.array(
            [np.repeat(0, self._width) for i in range(self._height)]),
                                   columns=[i for i in range(self._width)])
        print(self._board)
        self._lines = 0
        self._score = 0
        # center of the current tetrimino
        self._cur_pos = [self._width / 2, self._height]
        self._cur = Tetrimino.gen().points()
        self._next = Tetrimino.gen().points()

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
                    self._adjust(event.key)
            now = time.time()
            if now - last >= 1 / self._speed:
                print('dropping 1')
                print(self._cur_pos)
                if not self._drop_one():
                    if not self._freeze():
                        run = False
                last = now
            self._update()

        pygame.quit()

    def _adjust(self, key):
        print(pygame.key.name(key))
        if key == pygame.K_UP:
            self._rotate()
        elif key == pygame.K_DOWN:
            self._drop_one()
        elif key == pygame.K_LEFT:
            self._move_h(-1)
        elif key == pygame.K_RIGHT:
            self._move_h(1)

    def _min_x(self, cur=None):
        if cur is None:
            cur = self._cur
        return np.amin(cur, axis=0)[0] - 0.5

    def _max_x(self, cur=None):
        if cur is None:
            cur = self._cur
        return np.amax(cur, axis=0)[0] + 0.5

    def _min_y(self, cur=None):
        if cur is None:
            cur = self._cur
        return np.amin(cur, axis=0)[1] - 0.5

    def _rotate(self):
        # we rotate 90 degrees at a time, just play the trick
        cur = np.array([(y, -x) for x, y in self._cur])
        if self._will_collide(cur, self._cur_pos):
            return

        min_x = self._min_x(cur)
        if self._cur_pos[0] + min_x < 0:
            if not self._move_h(-min_x - self._cur_pos[0]):
                return
        max_x = self._max_x(cur)
        if self._cur_pos[0] + max_x > self._width:
            if not self._move_h(-max_x - self._cur_pos[0] + self._width):
                return
        self._cur = cur
        print(f'cur pos {self._cur_pos}')

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
            ) + by <= self._width and not self._will_collide(
                    self._cur, [self._cur_pos[0] + by, self._cur_pos[1]]):
                self._cur_pos[0] += by
                return True
        return False

    def _drop_one(self) -> bool:
        if self._cur_pos[1] + self._min_y() > 0 and not self._will_collide(
                self._cur, [self._cur_pos[0], self._cur_pos[1] - 1]):
            self._cur_pos[1] -= 1
            return True
        return False

    def _freeze(self):
        for x, y in self._cur:
            self._board.at[int(self._cur_pos[1] + y - 0.5),
                           int(self._cur_pos[0] + x - 0.5)] = 1
        df = self._board
        df['sum'] = df.sum(axis=1)
        temp = df[df['sum'] == self._width]
        if not temp.empty:
            self._score += df['sum'].sum()
            df.drop(index=temp.index, inplace=True)
            for i in range(len(temp.index)):
                df.loc[-1 * i - 1] = 0
            df.index = np.arange(0, self._height, 1)
            print(self._board)
        self._board.drop(columns=['sum'], inplace=True)
        self._cur = self._next
        self._next = Tetrimino.gen().points()
        self._cur_pos = [self._width / 2, self._height]
        return not self._will_collide(self._cur, self._cur_pos)

    def _will_collide(self, points, cur_pos):
        for x, y in points:
            y = int(cur_pos[1] + y - 0.5)
            x = int(cur_pos[0] + x - 0.5)
            if y < 0 or y >= self._height or x < 0 or x >= self._width:
                continue
            if self._board.at[y, x] == 1:
                return True
        return False

    def _update(self):
        self._window.fill(0)

        for y in range(self._height):
            for x in range(self._width):
                if self._board.at[y, x] == 1:
                    pygame.draw.rect(self._window, (255, 255, 255),
                                     self._rect(x + 0.5, y + 0.5))
        for r in self._cur:
            pygame.draw.rect(
                self._window, (255, 255, 255),
                self._rect(r[0] + self._cur_pos[0], r[1] + self._cur_pos[1]))
        pygame.display.flip()

    def _rect(self, x, y):
        rect = pygame.Rect(0, 0, _RECT_LEN, _RECT_LEN)
        rect.center = (x * _RECT_LEN, (self._height - y) * _RECT_LEN)
        return rect
