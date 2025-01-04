import yaml
import datetime
import json
import os
import numpy as np
from cerberus import Validator

from simulation.entities import Entities
from utils.numpy_encoder import NumpyEncoder
from utils.create_path_plots import create_path_plots
import config.validation_templates as templates


# Define the maximum number of entities that can spawn at the same time.
# Dead bullet entities will not count towards the maximum. Meaning the
# chances of going beyond this limit are very slim.
# !!! CAUTION !!!: 
#   Increasing this number may have significant impact on runtime!
MAX_ENTITIES = 1000

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
        plane_config: str="config/i-16_falangist.yaml",
        env_config: str="config/default_env.yaml",
        target_config: str="config/default_target.yaml",
        seed: int=42
    )-> None:
        """
        Initializer for BaseEnv class.

        @params:
            - plane_config (str): Path to yaml file with plane 
            configuration. See config/i-16_falangist.yaml for more info.
            - env_config (str): Path to yaml file with environment 
            configuration. See config/default_env.yaml for more info.
            - target_config (str): Path to yaml file with target 
            configuration. See config/default_target.yaml for more 
            info.
            - seed (int): seed for randomizer 
        """
        np.random.seed(seed)

        # for saving the observation history, used in self.close()
        self._current_iteration = 0
        self._observation_history = {self._current_iteration : []}

        # delta with which to update the environment each tick
        self._dt = 1 / 60
        
        # validate all of the provided config files
        validator = Validator()
        with open(plane_config, 'r') as stream:
            self._plane_data = yaml.safe_load(stream)
        assert validator.validate(
            self._plane_data, 
            templates.PLANE_TEMPLATE
        ), f"A validation error occurred in the plane data: {validator.errors}"

        with open(env_config, 'r') as stream:
            self._env_data = yaml.safe_load(stream)
        assert validator.validate(
            self._env_data, 
            templates.ENVIRONMENT_TEMPLATE
        ), f"A validation error occurred in the env data: {validator.errors}"

        with open(target_config, 'r') as stream:
            self._target_data = yaml.safe_load(stream)
        assert validator.validate(
            self._target_data, 
            templates.TARGET_TEMPLATE
        ),f"A validation error occurred in the target data: {validator.errors}"

        # reserve memory for necessary member objects
        self._entities = None
        
        self._create_entities()

    def _create_entities(self)-> None:
        """
        Creates plane and target entities.

        Uses self._create_agent() and self._create_target() to do this.
        Combines scalars of these two objects into two big matrices, 
        which it uses to create an Entities object.
        """
        agent_scalars, agent_vectors = self._create_agent()
        target_scalars, target_vectors = self._create_target()
        scalars = np.array([agent_scalars, target_scalars])

        vectors = np.array([agent_vectors, target_vectors])

        window_dimensions = self._env_data["window_dimensions"]

        boundaries = np.array(
            [
                [0,  window_dimensions[0]],
                [0,  window_dimensions[1]]
            ]
        )

        self._entities = Entities(scalars, vectors, MAX_ENTITIES, boundaries)

    def _create_agent(self)-> tuple[np.ndarray, np.ndarray]:
        """
        Create agent object for self.

        Use plane data to create Plane object.
        
        @returns:
            - tuple with numpy arrays containing scalars and vectors
        """

        scalars = np.array(list(self._plane_data["properties"].values())[:10])
        # the extra data is [aoa_degree, entity_type, coll_flag, debug]
        scalars = np.concatenate((scalars, np.array([0, 0, -1, 0])))

        vectors = np.array(list(self._plane_data["properties"].values())[10:])
        # The extra data is
        # [v_uv, f_gravity, f_engine, f_drag, f_lift, pitch_uv]
        vectors = np.concatenate((vectors, np.zeros(shape=(6,2), dtype=float)))

        return scalars, vectors

    def _create_target(self)-> tuple[np.ndarray, np.ndarray]:
        """
        Create target object for self.

        Use target data to create Target object.

        @returns:
            - tuple with numpy arrays containing scalars and vectors
        """
        scalars = np.array(list(self._target_data.values())[3])
        # the only data needed is the collision radius
        scalars = np.concatenate(
            (
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                np.array([scalars]),
                np.array([0, 1, -1, 0])
            )
        )

        vectors = np.array(list(self._target_data.values())[2])
        # the only data needed is the position
        vectors = np.concatenate(
            (
                np.zeros(shape=(3,2), dtype=float),
                np.array([vectors]),
                np.zeros(shape=(6,2), dtype=float)
            )
        )

        return scalars, vectors

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
        direction_to_target = self._entities.targets.vectors[0, 3] - state[:2]
        
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
        return np.all(self._entities.targets.scalars[:, 12] != -1)
    
    def _check_if_truncated(self)-> bool:
        """
        Check if the current conditions result in a truncated state.

        Truncated state is defined as a state where the agent crashes.
        For example, the agent can crash into the ground.

        @returns:
            - boolean; True if truncated, False if not
        """
        return np.all(self._entities.airplanes.scalars[:, 12] != -1)

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
        pos = self._entities.airplanes.vectors[0, 3]
        v = self._entities.airplanes.vectors[0, 2]
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

        Performs action on agent.

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
        actions = np.array([[0, action]])
        self._entities.tick(self._dt, actions)

        # calculate, save, and return observation in current conditions
        observation = self._calculate_observation()
        self._observation_history[self._current_iteration].append(observation)
        
        return observation[:1] + \
            (observation[1] - 50 if action == 5 else observation[1],) + \
            observation[2:]

    def reset(self, seed: int=42)-> tuple[np.ndarray, dict]:
        """
        Reset environment.

        Will create completely new entities.
        Adds new page to the history dictionary.
        Returns initial state & info.

        @params:
            - seed (int): seed used to spawn in the entities.
        
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
        pos = self._entities.airplanes.vectors[0, 3]
        v = self._entities.airplanes.vectors[0, 2]
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
        if save_figs:
            create_path_plots(
                folder_path, 
                self._observation_history, 
                self._env_data,
                figs_stride
            )
