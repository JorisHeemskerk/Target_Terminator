import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def create_path_plots(
    folder_path: str, 
    observation_history: dict,
    env_data: dict,
)-> None:
    """
    Create plots that display the path of the agent.

    It tries to draw background and floor of the environment on the 
    figure. Above which it plots the x,y flight history of the agent.
    It colours this graph in accordance with the normalized reward 
    provided. It saves the figure in the provided folder. It does this
    for each of the runs in the observation history.

    @params:
        - folder_path (str): Path to output folder.
        If this folder does not exist, no new one will be made.
        - observation_history (dict): Dictionary containing list of 
        observations per iteration/run.
        - env_data (dict): Environment configuration.
            See config/default_env.yaml for more info.
            In theory, it only needs to contain the window dimensions
            and preferably the background and floor data.
    """
    for iteration, observations in observation_history.items():
        vertices = [(x, y, r) for (x, y, _, _), r, _, _, _ in observations]
        
        xs, ys, rewards = zip(*vertices)
        normalise_rewards = plt.Normalize(min(rewards), max(rewards))

        colour_map = plt.get_cmap('RdYlGn')

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a LineCollection from the segments
        lc = LineCollection(segments, cmap=colour_map, norm=normalise_rewards)
        lc.set_array(np.array(rewards))
        
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.set_xlim(0, env_data["window_dimensions"][0])
        ax.set_ylim(0, env_data["window_dimensions"][1])
        ax.invert_yaxis()
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Reward')
        ax.set_title(f"Flight path for iteration {iteration}.")

        # try to plot the backgrounds, if available
        # if any of these settings are missing, nothing will be plotted
        try:
            background_image = plt.imread(env_data["background"]["sprite"])
            ax.imshow(background_image)

            ground_image = plt.imread(env_data["ground"]["sprite"])
            ax.imshow(ground_image, extent=[
                0, # left
                env_data["window_dimensions"][0], # right
                env_data["window_dimensions"][1], # bottom
                env_data["ground"]["collision_elevation"], #top
            ])
        except KeyError:
            pass

        plt.savefig(f"{folder_path}/flight_path_it-{iteration}")
        plt.close(fig)
