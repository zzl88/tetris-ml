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

    # play
    # game.Game().play()

    # train
    a = agent.Agent(gamma=0.9,
                    output_size=5,
                    memory_size=5000,
                    learning_rate=0.0001)
    try:
        a.load_weights('weights3')
    except:
        pass
    for i in range(100000):
        g = game.Game(speed=50)
        reward_acc = g.train(a, batch_size=250)
        LOGGER.error(
            f'iteration[{i}] score[{g.score()}] reward_acc[{reward_acc}]')
        if (i + 1) % 100 == 0:
            LOGGER.error(f'saving')
            a.save_weights('weights3')
