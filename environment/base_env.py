# import yaml
import datetime
import json
import os
import numpy as np
import pygame
from cerberus import Validator

from simulation.entities import Entities
from utils.numpy_encoder import NumpyEncoder
from utils.create_path_plots import create_path_plots
# import config.validation_templates as templates
# import utils.collision as col


class BaseEnv():
    """
    Base environment class.

    This class instantiates an entire environment, excluding a GUI.
    It creates the environment, plane, and target, as stated in the
    provided config files.

    This class has no public member variables.

    @public methods:
    + step(action: int)-> np.ndarray
        Takes a step in the environment. This means that the plane
        will be updated based on the action taken and that the 
        environment will react accordingly.
    + reset(seed: int=42)-> tuple[np.ndarray, dict]
        Resets the environment given a seed. This means that the plane
        and target will be reset to their spawn locations.
    + close()-> None
        Closes the environment and thereby outputs its entire history.
    """
    def __init__(
        self, 
        seed: int=42
    )-> None:
        np.random.seed(seed)

        # for saving the observation history, used in self.close()
        self._current_iteration = 0
        self._observation_history = {self._current_iteration : []}

        # self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        self._dt = 0
        
        self._create_entities()

    def _create_entities(self)-> None:
        scalars = np.array(
            #   0       1       2       3       4       5       6       7       8   9   10  11  12  13
            [[  1200,   0.6,    100,    0.32,   0.5,    300,    100,    100,    0,  6,  0,  0,  -1, 0], 
            [  0,      0,      0,      0,      0,      0,      0,      0,      0,  10, 0,  1,  -1, 0]]
        )

        vectors = np.array(
            #   0               1               2       3             4       5       6       7       8       9
            [[  [-15.0, -0.95], [19.0, 1.4],    [100,0],[100, 300],   [0,0],  [0,0],  [0,0],  [0,0],  [0,0],  [0,0]],
            [  [0,0],          [0,0],          [0,0],  [800, 500],    [0,0],  [0,0],  [0,0],  [0,0],  [0,0],  [0,0]]]
        )

        boundaries = np.array(
            [
                [0,  1280   ],
                [0,  720 ]
            ]
        )

        self.entities = Entities(scalars, vectors, 1000, boundaries)

    def _calculate_reward(self, state: np.ndarray)-> float:
        """
        Reward function for environment.

        Reward is equal to the difference between the unit vector from 
        the plane to the target and the unit vector for the plane's 
        velocity will be subtracted.

        NOTE: function does not check for validity of state parameter

        @params:
            - state (np.ndarray): 
            state contains:
                * x (float): x position of plane
                * y (float): y position of plane
                * velocity_x (float): velocity of plane in x direction
                * velocity_y (float): velocity of plane in y direction
        
        @returns:
            - float with reward.
        """
        direction_to_target = self.entities.targets.vectors[:, 3][0] - \
            state[:2]
        
        unit_vector_to_target = direction_to_target / \
            np.linalg.norm(direction_to_target)

        velocity = state[2:4]
        unit_vector_agent = velocity / np.linalg.norm(velocity)

        return -50 * np.linalg.norm(unit_vector_agent - unit_vector_to_target) 
    
    def _check_if_terminated(self)-> bool:
        """
        Check if the current conditions result in a terminal state.

        Terminal state is defined as a state where a bullet collides
        with the target.

        @returns:
            - boolean; True if terminal, False if not
        """
        return np.all(self.entities.targets.scalars[:, 12] != -1)
    
    def _check_if_truncated(self)-> bool:
        """
        Check if the current conditions result in a truncated state.

        Truncated state is defined as a state where the agent crashes.
        For example, the agent can crash into the ground.

        @returns:
            - boolean; True if truncated, False if not
        """
        return np.all(self.entities.airplanes.scalars[:, 12] != -1)

    def _calculate_observation(
            self
        )-> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Calculate observation of current conditions.

        Observation consists of:
            - state, which contains:
                * x (float): x position of plane
                * y (float): y position of plane
                * velocity_x (float): velocity of plane in x direction
                * velocity_y (float): velocity of plane in y direction
            - reward (see self._calculate_reward())
                terminal states are rewarded a bonus of 1,000,000, 
                whilst truncated states are rewarded with -1,000,000,000
            - is_terminal (see self._check_if_terminal)
            - is_truncated (see self._check_if_truncated)
            - info, made for compatibility with Gym environment, 
            but is always empty.

        @returns:
             - np.ndarray with state
             - float with reward
             - bool with is_terminal
             - bool with is_truncated
             - dict with info (always empty)
        """
        pos = self.entities.airplanes.vectors[:, 3]
        v = self.entities.airplanes.vectors[:, 2]
        state = np.concatenate((pos, v), axis=None)
        
        is_terminated = self._check_if_terminated()
        is_truncated = self._check_if_truncated()
        reward = self._calculate_reward(state)
        
        if is_terminated:
            reward += 200
        if is_truncated:
            reward -= 1_000

        return(state, reward, is_terminated, is_truncated, {})

    def _render(self)-> None:
        """
        Render function for all of the graphical elements of the 
        environment.

        Since the base class does not have any gui elements, this method
        is not implemented here.
        """
        raise NotImplementedError(
            "As this class has no gui, this is not implemented."
        )

    def step(self, action: int)-> np.ndarray:
        """
        Step function for environment.

        Performs action on self._agent.

        @params:
            - action (int): one of:
                * 0: do nothing
                * 1: adjust pitch upwards
                * 2: adjust pitch downwards
                * 3: increase throttle
                * 4: decrease throttle
                * 5: shoot a bullet
        
        @returns:
            - np.ndarray with observation of resulting conditions
        """
        actions = np.array([[0,action]])
        self.entities.tick(self._dt, actions)

        self._dt = self.clock.tick(60) / 1000

        # calculate, save, and return observation in current conditions
        observation = self._calculate_observation()
        self._observation_history[self._current_iteration].append(observation)
        
        return observation[:1] + \
            (observation[1] - 50 if action == 5 else observation[1],) + \
            observation[2:]

    def reset(self, seed: int=42)-> tuple[np.ndarray, dict]:
        """
        Reset environment.

        Will create completely new agent and target.
        Adds new page to the history dictionary.
        Returns initial state & info.

        @params:
            - seed (int): seed used to spawn in the agent and target.
        
        @returns:
            - np.ndarray with initial state 
            (see self._calculate_observation())
            - dict with info, made for compatibility with Gym 
            environment, but is always empty.
        """
        np.random.seed(seed)
        self._create_entities()

        self._current_iteration += 1
        self._observation_history[self._current_iteration] = []

        # the agent's current coordinates are defined by the centre of 
        # its rect
        pos = self.entities.airplanes.vectors[:, 3]
        v = self.entities.airplanes.vectors[:, 2]
        return np.concatenate((pos, v), axis=None), {}

    def close(
        self, 
        save_json: bool=False, 
        save_figs: bool=False, 
        figs_stride: int=1
    )-> None:
        """
        Close environment and output history.

        Will create a folder indicated by the current date and time, 
        provided save == True in which resides:
            - a json file with the entire observation history.
            - an image per iteration, which displays the flown path of 
            the agent, along with the reward (indicated by the colour).
        
        @params:
            - save_json (bool): Save json or not.
            - save_figs (bool): Save the plots or not.
            - figs_stride (int): Stride for saving the figures.
        """
        # prepare the output folder
        if save_json or save_figs:
            folder_path = "output/" \
            f"{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}"
            os.mkdir(folder_path)

        # write all the observations to a json file
        if save_json:
            with open(
                f"{folder_path}/_observation_history.json", "w"
            ) as outfile: 
                json.dump(self._observation_history, outfile, cls=NumpyEncoder)

        # create all the graphs and save them to the `folder_path`
        # if save_figs:
        #     create_path_plots(
        #         folder_path, 
        #         self._observation_history, 
        #         self._env_data,
        #         figs_stride
        #     )
