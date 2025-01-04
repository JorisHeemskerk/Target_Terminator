import copy
from tqdm import tqdm
import numpy as np
import time
from simulation.entities import Entities
import random



def run(entities):
    l = [0, 1, 2, 3, 4, 5]
    s = entities.scalars
    v = entities.vectors
    b = entities.boundaries

    running = True
    dt = 0
    # center = np.array((screen.get_width() / 2, screen.get_height() / 2))

    airplane_sprites = []
    target_sprites = []
    bullet_sprites = []
    t = time.time()
    for j in tqdm(range(1000)):
        # e = copy.copy(entities)
        e = Entities(s.copy(), v.copy(), 1000, b)
        for i in range(100_000):

            a = random.choice(l)
            # actions = []

            actions = np.array([[0,a]])
            e.tick(dt,actions)
            if(e.scalars[0,12] != -1):
                break

            # print(entities.scalars[0,12])
            # print(e.vectors[0,3])


            # dt = clock.tick(60) / 1000
            dt = 1/60

    print(i)
    print(t-time.time())

    # pygame.quit()