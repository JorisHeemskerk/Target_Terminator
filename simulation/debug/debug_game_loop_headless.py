import time
import numpy as np
from tqdm import tqdm
import random
import copy

dt = 1/60
l = [0,1,2,3,4,5]

def run(entities):
    t = time.time()
    for _ in tqdm(range(100_000)):
        e = copy.deepcopy(entities)
        for n in range(100_000):
            a = random.choice(l)
            actions = np.array([[0,a]])

            e.tick(dt, actions)
            # print(dt)
            # print(e.scalars[:4])
            # print(e.vectors[:4])

            if(e.scalars[0,12] != -1):
                print(n)
                break

    print(time.time()-t)
