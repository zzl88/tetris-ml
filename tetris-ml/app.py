import logging

import game
import agent

LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    formatter = logging.Formatter(
        '%(levelname)s %(asctime)s %(pathname)s:%(lineno)d - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.ERROR)

    # play
    # game.Game().play()
    # exit()

    # train
    a = agent.Agent(gamma=0.9,
                    output_size=40,
                    memory_size=5000,
                    learning_rate=0.00015).to(agent.DEVICE)
    try:
        pass
        a.load_weights('weights4')
    except:
        pass
    for i in range(200000):
        g = game.Game(speed=0)
        reward_acc, train_size = g.train(a, batch_size=250)
        LOGGER.error(
            f'iteration[{i}] score[{g.score()}] reward_acc[{reward_acc}] size[{train_size}]'
        )
        if (i + 1) % 300 == 0:
            LOGGER.error(f'saving')
            a.save_weights('weights4')
