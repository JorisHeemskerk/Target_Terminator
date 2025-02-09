# Target Terminator
By *Bas de Blok, Finn de Graaf* & *Joris Heemskerk*

![TT_readme_intro_png](assets/TT_readme_intro.png)

Created for the Adaptive Systems cursus at the University of applied sciences Hogeschool Utrecht. 

Target Terminator is a project for single agent reinforcement learning.

## Installation guide

1. Downloading this repository\
We recommend downloading this repository as a .zip file and extracting it into your current project rather than cloning the repository, as this may result in conflicts when working with nested git folder structures. You can download the .zip file in the top right on the [main page](https://github.com/JorisHeemskerk/Target_Terminator) of this repository, using the `<> Code ▾` button.\
Place this directory in your current project. The folder structure should look something like this:
    ```
    your_project/
    │
    ├─ Target_Terminator/
    │
    ├─ your_code_files.py
    ```
2. New Conda environment.\
    Make a new Conda environment in any terminal using:
    ```shell
    conda create -n "AS_TT" python=3.11.9
    ```
    What exactly you name this environment does not matter.
3. Activate conda environment with:
    ```shell
    conda activate AS_TT
    ```
4. Install the requirements using:
    ```shell
    pip install -r Target_Terminator/requirements.txt
    ```

## Hello world!

Target Terminator allows for 3 different modes to simulate the environment:
- `keyboard` = allows you to fly using your arrow keys and fire bullets by pressing the spacebar.
- `human` = uses the actions determined by your programme to move the plane.
- `base` = same as human but without the gui, allows for quicker training.

This Hello world code snippet allows you to check the workings of all these modes.
```py
import Target_Terminator as TT
import random

env = TT.make(render_mode="keyboard")
# env = TT.make(render_mode="human")
# env = TT.make(render_mode="base")

for _ in range(10000):
    # in keyboard mode the random.choice will be ignored
    state, reward, terminated, truncated, _ = env.step(random.choice([1,2,3,4,5]))

    # respawn agent when crashed or won
    if truncated or terminated:
        env.reset()
env.close()
```

## Config files

In the Target Terminator folder you'll find a config folder containing multiple .yaml files. If you want to use different settings for your Target Terminator environment you can do so by creating new .yaml files in this folder and using your own parameters or png's. \
**NOTE:** See validation_templates/ for the parameter restrictions. 

To use these newly created .yaml files you'll need to add them as function parameters in the make function. \
Example code:
```py
env = TT.make(
    render_mode="human",
    plane_config="config/new_plane.yaml",
    env_config="config/new_env.yaml",
    target_config="config/new_target.yaml"
)
```