from simulation.entities import Entities
import debug_game_loop
import numpy as np

scalars = np.array(
    #   0       1       2       3       4       5       6       7       8   9   10  11  12  13
    [[  1200,   0.6,    100,    0.32,   0.5,    300,    100,    100,    0,  6,  0,  0,  -1, 0],
     [  1200,   0.6,    100,    0.32,   0.5,    300,    100,    100,    0,  6,  0,  0,  -1, 0],
     [  0,      0,      0,      0,      0,      0,      0,      0,      0,  10, 0,  1,  -1, 0],
     [  0,      0,      0,      0,      0,      0,      0,      0,      0,  10, 0,  1,  -1, 0]]
)


vectors = np.array(
    #   0               1               2       3           4       5       6       7       8       9
    [[  [-15.0, -0.95], [19.0, 1.4],    [0,0],  [-200,0],   [0,0],  [0,0],  [0,0],  [0,0],  [0,0],  [0,0]],
     [  [-15.0, -0.95], [19.0, 1.4],    [3,4],  [-200,-100],[0,0],  [0,0],  [0,0],  [0,0],  [0,0],  [0,0]],
     [  [0,0],          [0,0],          [0,0],  [0,0],      [0,0],  [0,0],  [0,0],  [0,0],  [0,0],  [0,0]],
     [  [0,0],          [0,0],          [0,0],  [-200,16.01],   [0,0],  [0,0],  [0,0],  [0,0],  [0,0],  [0,0]]]
)

actions = np.array([])

entities = Entities(scalars, vectors, 1000)
# entities.tick(0, actions)

debug_game_loop.run(entities)
