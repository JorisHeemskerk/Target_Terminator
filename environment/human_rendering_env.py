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
        self,
        plane_config: str="config/i-16_falangist.yaml",
        env_config: str="config/default_env.yaml",
        target_config: str="config/default_target.yaml",
        seed: int=42
    )-> None:
        """
        Initializer for HumanRenderingEnv class.

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
        # place pygame window in top left of monitor(s)
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{0},{0}"
        pygame.init()
            
        super().__init__(
            plane_config=plane_config,
            env_config=env_config,
            target_config=target_config,
            seed=seed
        )

        # sprite data is not mandatory in config, 
        # so we check these here
        assert "sprite" in self._plane_data and \
            "side_view_dir" in self._plane_data["sprite"] and \
            "top_view_dir" in self._plane_data["sprite"], \
            "Either `sprite`, `sprite : side_view_dir`, or `sprite : "\
            "top_view_dir` are not present in plane data."
        assert "sprite" in self._target_data, "`sprite` key not in target data"
        assert "sprite" in self._env_data["ground"], \
            "`sprite` key not in `ground` field in target data"
        assert "sprite" in self._env_data["background"], \
            "`sprite` key is not in background field in target data"
        
        self.screen = pygame.display.set_mode(
            self._env_data["window_dimensions"]
        )
        
        pygame.display.set_caption('Target terminator')

        self._create_sprites()

    def _create_sprites(self)-> None:
        """
        Create background object for self.

        Use environment data to create background object.
        """
        self._background_sprite = pygame.image.load(
            self._env_data["background"]["sprite"]
        )
        self._background_sprite = pygame.transform.scale(
            self._background_sprite,
            pygame.display.get_surface().get_size()
        )

        self._target_sprite = pygame.transform.scale(
            pygame.image.load(self._target_data["sprite"]), 
            self._target_data["size"]
        )

        self._bullet_sprite = pygame.transform.scale(
            pygame.image.load(self._plane_data["bullet_config"]["sprite"]),
            self._plane_data["bullet_config"]["size"]
        )

        self._plane_sprite = pygame.transform.scale(
            pygame.image.load(self._plane_data["sprite"]["side_view_dir"]),
            self._plane_data["sprite"]["size"]
        )

    def _render(self) -> None:
        """
        Render function for all of the graphical elements of the 
        environment.
        """
        # gather all rotation instructions for bullets and save to tuple
        alive_bullets = self._entities.bullets.vectors[
            self._entities.bullets.scalars[:, 11] != -1
        ]
        rotate_instructions = np.degrees(
            np.arctan2(alive_bullets[:, 2, 0], alive_bullets[:, 2, 1]) + 270
        ) % 360

        blit_data_bullets = []
        for bullet_vectors, rotate_instruction in zip(
            alive_bullets, 
            rotate_instructions
        ):
            rotated_sprite = pygame.transform.rotate(
                self._bullet_sprite,
                rotate_instruction
            )
            # use coordinates as center for sprite
            bullet_rect = rotated_sprite.get_rect()
            bullet_rect.center = bullet_vectors[3]
            blit_data_bullets.append((rotated_sprite, bullet_rect.topleft))

        # gather all rotation instructions for planes and save to tuple
        alive_airplanes = self._entities.airplanes.vectors[
            self._entities.airplanes.scalars[:, 11] != -1
        ]
        rotate_instructions = self._entities.airplanes.scalars[
            self._entities.airplanes.scalars[:, 11] != -1
        ][:, 8]

        blit_data_planes = []
        for airplane_vectors, rotate_instruction in zip(
            alive_airplanes, 
            rotate_instructions
        ):
            rotated_sprite = pygame.transform.rotate(
                self._plane_sprite,
                rotate_instruction
            )
            # use coordinates as center for sprite
            plane_rect = rotated_sprite.get_rect()
            plane_rect.center = airplane_vectors[3]
            blit_data_planes.append((rotated_sprite, plane_rect.topleft))

        # put target sprite position in center
        target_rect = self._target_sprite.get_rect()
        target_rect.center = self._entities.targets.vectors[:, 3][0]

        # blit all objects in order of background, target, bullet, plane
        self.screen.blits(
            blit_sequence=[
                # Background
                (self._background_sprite, (0, 0)),
                # Target
                (self._target_sprite, target_rect.topleft),
            ] + blit_data_bullets + blit_data_planes
        )
        
        pygame.display.flip()

        
    def step(self, action: int)-> np.ndarray:
        """
        Step function for environment.

        Performs action on agent and renders frame.

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

        Will create completely new entities.
        Adds new page to the history dictionary
        return initial state & info. Renders the initial frame

        @params:
            - seed (int): seed used to spawn in the entities.
        
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
