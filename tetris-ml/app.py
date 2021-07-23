import datetime
import logging

import game
import agent

LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    formatter = logging.Formatter(
        '%(asctime)s %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.ERROR)

    # game.Game().run()
    a = agent.Agent(gamma=0.9,
                    layers=[200, 2000, 800, 3000, 40],
                    memory_size=5000,
                    learning_rate=0.0001)
    a.load_weights('weights3')
    for i in range(100000):
        g = game.Game(speed=0)
        reward_acc = g.train(a, batch_size=250)
        LOGGER.error(
            f'iteration[{i}] score[{g.score()}] reward_acc[{reward_acc}]')
        if (i + 1) % 1000 == 0:
            a.save_weights('weights3')
