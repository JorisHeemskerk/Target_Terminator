import numpy as np
import time


class Bullets:
    def __init__(self, scalars, vectors):
        self.scalars = scalars
        self.vectors = vectors
        self.n_bullets = 0

    def tick(self, dt):
        # self.vectors[:, 3] += dt * self.vectors[:, 2]
        self.vectors[:self.n_bullets, 3] += dt * self.vectors[:self.n_bullets, 2]

        # TODO: het is me zo niet duidelijk of het sneller is om *alle* rijen
        #  gereserveerd voor bullets, dan wel niet in gebruik, te updateten, of
        #  dat het sneller is eerst n_bullets te updateten en vervolgens
        #  *alleen* de rijen met bestaande bullets te updateten. als dit tweede
        #  het geval is gebruiken we dus (de weg-gecommentete) line 9 ipv 8

    # def spawn(self, vectors):
    #     scalars = np.zeros((vectors.shape[0], self.scalars.shape[1]))
    #     scalars[:,11] = 2
    #     scalars[:,12] = -1
    #     scalars[:,2] = time.time()
    #     i = np.argsort(self.scalars[:,11])[:scalars.shape[0]]
    #     self.scalars[i] = scalars
    #     self.vectors[i] = vectors

    def spawn(self, vectors):
        n = vectors.shape[0]
        self.n_bullets += n
        self.scalars[n:] = self.scalars[:-n]
        self.vectors[n:] = self.vectors[:-n]
        self.scalars[:n, 11] = 2
        self.scalars[:n, 12] = -1
        self.scalars[:n, 2] = 0
        self.vectors[:n] = vectors

    def despawn(self):
        self.scalars[:self.n_bullets, 2] += 1
        n = np.count_nonzero(self.scalars[:self.n_bullets, 2] > 180)
        self.scalars[self.n_bullets-n:self.n_bullets, 11] = -1
        self.n_bullets -= n
        return n

