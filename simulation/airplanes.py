import numpy as np
import math
from sklearn.preprocessing import normalize

class Airplanes:
    def __init__(self, scalars, vectors):
        self.scalars = scalars
        self.vectors = vectors
        self.vectors[:,5,1] = self.scalars[:,0] * 9.81

    def tick(self, dt, actions):
        # update pitch unit vector
        # TODO: dit kan fout gaan als pitch exact 0 is, ik weet niet zo goed of
        #  dat mogelijk is en hoe erg het dan fout gaat, maar iig misschien
        #  beter om hier iets met atan2 te doen ofzo
        self.vectors[:, 9, 0] = np.cos(-math.pi / 180 * self.scalars[:, 8])
        self.vectors[:, 9, 1] = np.sin(-math.pi / 180 * self.scalars[:, 8])

        # update velocity unit vector
        self.vectors[:, 4] = normalize(self.vectors[:, 2])

        # update AoA
        self.scalars[:, 10] = ((
            np.arctan2(self.vectors[:, 9, 0], self.vectors[:, 9, 1]) -
            np.arctan2(self.vectors[:, 2, 0], self.vectors[:, 2, 1])
        ) * 180 / math.pi + 180) % 360 - 180

        # engine force vector
        self.vectors[:, 6] = self.scalars[:, 7][:, None] * 0.1 * self.scalars[:, 5][:, None] * self.vectors[:, 9]

        # lift force vector
        coef_lift = self._lift_curve()
        norm_lift = self.scalars[:, 2] * coef_lift * np.linalg.norm(self.vectors[:, 2], axis=1) ** 2
        self.vectors[:, 8, 0] = norm_lift * self.vectors[:, 4, 1]
        self.vectors[:, 8, 1] = norm_lift * -self.vectors[:, 4, 0]

        # drag force vector
        coef_drag = (self.scalars[:, 10] / np.sqrt(40))**2 + self.scalars[:, 4]
        norm_drag = self.scalars[:, 1] * coef_drag * np.linalg.norm(self.vectors[:, 2], axis=1)**2
        self.vectors[:, 7] = -norm_drag[:, None] * self.vectors[:, 4]

        # fres + update v, pos
        f_res = self.vectors[:, 5] + self.vectors[:, 6] + self.vectors[:, 7] + self.vectors[:, 8]
        self.vectors[:, 2] += dt * f_res / self.scalars[:, 0][:, None]
        self.vectors[:, 3] += dt * self.vectors[:, 2]

        # induced torque
        filter = np.zeros(self.scalars.shape[0])
        filter[self.scalars[:, 10] < self.vectors[:, 0, 0]] = 1
        filter[self.scalars[:, 10] > self.vectors[:, 1, 0]] = -1
        self.scalars[:, 8] = (self.scalars[:, 8] + dt * filter * norm_drag * 0.01) % 360

        self.execute_actions(dt, actions)

    def execute_actions(self, dt, actions):
        l = self.scalars.shape[0]
        self.scalars[0, 13] = actions[0, 1]

        # action 1 pitch up
        filter = np.zeros(l)
        filter[actions[actions[:, 1] == 1][:, 0]] = 1
        self.scalars[:, 8] = (self.scalars[:, 8] + self.scalars[:, 6] * dt * filter) % 360

        # action 2 pitch down
        filter = np.zeros(l)
        filter[actions[actions[:, 1] == 2][:, 0]] = 1
        self.scalars[:, 8] = (self.scalars[:, 8] - self.scalars[:, 6] * dt * filter) % 360

        # action 3 throttle up
        filter = np.zeros(l)
        filter[actions[actions[:, 1] == 3][:, 0]] = 1
        self.scalars[:, 7] += dt * 100 * filter
        self.scalars[self.scalars[:, 7] > 100, 7] = 100

        # action 4 throttle down
        filter = np.zeros(l)
        filter[actions[actions[:, 1] == 4][:, 0]] = 1
        self.scalars[:, 7] -= dt * 100 * filter
        self.scalars[self.scalars[:, 7] < 0, 7] = 0

        # TODO: action 6 flip

    def _lift_curve(self) -> np.ndarray:
        AoA = self.scalars[:, 10]
        AoA_crit_low = self.vectors[:, 0, 0]
        coef_low = self.vectors[:, 0, 1]
        AoA_crit_high = self.vectors[:, 1, 0]
        coef_high = self.vectors[:, 1, 1]
        cl0 = self.scalars[:, 3]

        lift_coef = np.zeros_like(AoA)

        # AoA < AoA_crit_low[0] - 1
        mask1 = AoA < (AoA_crit_low - 1)
        lift_coef[mask1] = 0.0

        # AoA_crit_low[0] - 1 <= AoA < AoA_crit_low[0]
        mask2 = (AoA >= (AoA_crit_low - 1)) & (AoA < AoA_crit_low)
        lift_coef[mask2] = coef_low[mask2] * np.abs(AoA_crit_low[mask2] - 1 - AoA[mask2])

        # AoA_crit_low[0] <= AoA < 0
        mask3 = (AoA >= AoA_crit_low) & (AoA < 0)
        b = cl0 - coef_low
        c = AoA / AoA_crit_low
        lift_coef[mask3] = cl0[mask3] - b[mask3] * c[mask3]

        # 0 <= AoA < AoA_crit_high[0]
        mask4 = (AoA >= 0) & (AoA < AoA_crit_high)
        b = coef_high - cl0
        c = AoA / AoA_crit_high
        lift_coef[mask4] = cl0[mask4] + b[mask4] * c[mask4]

        # AoA_crit_high[0] <= AoA < AoA_crit_high[0] + 1
        mask5 = (AoA >= AoA_crit_high) & (AoA < (AoA_crit_high + 1))
        lift_coef[mask5] = coef_high[mask5] * np.abs(AoA_crit_high[mask5] - 1 - AoA[mask5])

        # AoA >= AoA_crit_high[0] + 1
        mask6 = AoA >= (AoA_crit_high + 1)
        lift_coef[mask6] = 0.0

        return lift_coef
