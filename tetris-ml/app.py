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
    logging.root.setLevel(logging.INFO)

    # game.Game().run()
    a = agent.Agent(gamma=0.9,
                    layers=[201, 1000, 800, 2000, 100, 5],
                    memory_size=5000,
                    learning_rate=0.0001)
    # a.load_weights('weights')
    for i in range(500):
        g = game.Game(speed=0)
        reward_acc = g.train(a, batch_size=250)
        LOGGER.info(
            f'iteration[{i}] score[{g.score()}] reward_acc[{reward_acc}]')

    a.save_weights('weights')
