import numpy as np
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
    def __init__(self, speed=0.5):
        pygame.display.set_caption('Tetris')
        self._width = 10
        self._height = 20
        self._window = pygame.display.set_mode(
            (self._width * _RECT_LEN, self._height * _RECT_LEN))
        # 0 for empty, 1 for filled
        self._board = np.array(
            [np.repeat(0, self._width) for i in range(self._height)])
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
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    self._adjust(event.key)
            now = time.time()
            if now - last >= 2:
                print('dropping 1')
                print(self._cur)
                print(self._cur_pos)
                self._drop_one()
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
            if self._cur_pos[0] + self._min_x() >= 1:
                self._cur_pos[0] -= 1
        elif key == pygame.K_RIGHT:
            if self._cur_pos[0] + self._max_x() <= self._width - 1:
                self._cur_pos[0] += 1

    def _min_x(self):
        return np.amin(self._cur, axis=0)[0] - 0.5

    def _max_x(self):
        return np.amax(self._cur, axis=0)[0] + 0.5

    def _min_y(self):
        return np.amin(self._cur, axis=0)[1]

    def _max_y(self):
        return np.amax(self._cur, axis=0)[1]

    def _rotate(self):
        # we rotate 90 degrees at a time, just play the trick
        self._cur = np.array([(y, -x) for x, y in self._cur])
        min_x = self._min_x()
        if self._cur_pos[0] + min_x < 0:
            self._cur_pos[0] += -min_x - self._cur_pos[0]
        max_x = self._max_x()
        if self._cur_pos[0] + max_x > self._width:
            self._cur_pos[0] -= max_x + self._cur_pos[0] - self._width
        print(f'cur pos {self._cur_pos}')

    def _drop_one(self):
        self._cur_pos[1] -= 1

    def _update(self):
        self._window.fill(0)

        for x in range(self._width):
            for y in range(self._height):
                if self._board[y, x] == 1:
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
