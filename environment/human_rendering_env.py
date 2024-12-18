import pygame
import os
import numpy as np

from environment.base_env import BaseEnv


class HumanRenderingEnv(BaseEnv):
    """
    Base environment class.

    This class instantiates an entire environment, including a GUI.
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
        self
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
        """
        # place pygame window in top left of monitor(s)
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{0},{0}"
        pygame.init()
            
        super().__init__()
        
        self.screen = pygame.display.set_mode((1280, 720))
        
        pygame.display.set_caption('Target terminator')

        self._create_sprites()

    def _create_sprites(self)-> None:
        """
        Create background object for self.

        Use environment data to create background object.
        """
        self._background_sprite = pygame.image.load(
            "assets/background.png"
        )
        self._background_sprite = pygame.transform.scale(
            self._background_sprite,
            pygame.display.get_surface().get_size()
        )

        self._target_sprite = pygame.transform.scale(
            pygame.image.load("assets/target.png"), 
            [40, 40]
        )

        self._bullet_sprite = pygame.transform.scale(
            pygame.image.load("assets/bullet.png"),
            [10, 10]
        )

        self._plane_sprite = pygame.image.load(
                "assets/i16_falangist.png"
        )

        self._plane_sprite = pygame.transform.scale(
            self._plane_sprite,
            [24, 12]
        )

    def _render(self)-> None:
        """
        Render function for all of the graphical elements of the 
        environment.
        """
        self.screen.blit(self._background_sprite, (0, 0))
        # weggelaten, weet niet zeker of er rekening is gehouden met het bestaan van een vloer
        # self.screen.blit(self._floor.sprite, [0, self._floor.coll_elevation])
        self.screen.blit(
            self._target_sprite, 
            self.entities.targets.vectors[:, 3][0]
        )

        for bullet_vector, bullet_scalar in zip(
            self.entities.bullets.vectors,
            self.entities.bullets.scalars
            ):
            if bullet_scalar[11] == -1:
                continue
            self.screen.blit(
                pygame.transform.rotate(
                    self._bullet_sprite,
                    # From unit vector to radians to degrees
                    (
                        np.degrees(
                            np.arctan2(bullet_vector[2,0], bullet_vector[2,1])
                        ) + 270
                    ) % 360
                ),
                bullet_vector[3]
            )

        for plane_vector, plane_scalar in zip(
            self.entities.airplanes.vectors,
            self.entities.airplanes.scalars
        ):
            if plane_scalar[12] != -1:
                continue
            self.screen.blit(
                pygame.transform.rotate(
                    self._plane_sprite,
                    plane_scalar[8]
                ),
                plane_vector[3]
            )
        
        pygame.display.flip()
        
    def step(self, action: int)-> np.ndarray:
        """
        Step function for environment.

        Performs action on self._agent and renders frame.

        @params:
            - action (int): one of:
                * 0:  do nothing
                * 1: adjust pitch upwards
                * 2: adjust pitch downwards
                * 3: increase throttle
                * 4: decrease throttle
                * 5: shoot a bullet
        
        @returns:
            - np.ndarray with observation of resulting conditions
        """
        step_info = super().step(action=action)

        self._render()

        return step_info
    
    def reset(self, seed: int=42)-> tuple[np.ndarray, dict]:
        """
        Reset environment.

        Will create completely new agent and target.
        Adds new page to the history dictionary
        return initial state & info. Renders the initial frame

        @params:
            - seed (int): seed used to spawn in the agent and target.
        
        @returns:
            - np.ndarray with initial state 
            (see self._calculate_observation()).
            - dict with info, made for compatibility with Gym 
            environment, but is always empty.
        """
        output = super().reset(seed=seed)

        self._render()

        return output

    def close(
        self,
        save_json: bool=False, 
        save_figs: bool=False, 
        figs_stride: int=1
    )-> None:
        """
        Close environment and output history.

        Will create a folder indicated by the current date and time
        in which resides:
            - a json file with the entire observation history.
            - an image per iteration, which displays the flown path of 
            the agent, along with the reward (indicated by the colour).

        @params:
            - save_json (bool): Save json or not.
            - save_figs (bool): Save the plots or not.
        """
        pygame.display.quit()
        pygame.quit()
        super().close(
            save_json=save_json, 
            save_figs=save_figs,
            figs_stride=figs_stride
        )
