# *TODO:* UPDATE THIS THING


# Tupolev Gaming
By *Bas de Blok, Finn de Graaf* & *Joris Heemskerk*

Created for the [Autonomy By Design (ADB)](https://canvas.hu.nl/courses/39869/pages/kennisroute-ai) cursus.

## Theory
This simulation uses the multi agent theory from **MULTIAGENT SYSTEMS Algorithmic, Game-Theoretic, and Logical Foundations**, written by Yoav Shoham and Kevin Leyton-Brown. The theory from Chapter Two, Distributed Optimization has been implemented with the use of the Munkres library.

 During the simulations the teams try to shoot as many targets as possible. The teams assign a target to every agent on the team based on the *value* of the targets. The different strategies to calculate the value of a target are: 
 - Absolute distance (absolute_distance.py)
    - The shortest absolute distance between the target and the agent wins.
 - Absolute distance for two targets (two_targets_distance_team.py)
    - The shortest absolute distance between the target and two targets wins.
 - Energy required to reach the target (energy_bidding_team.py)
    - The least amount of adjusting required to reach the target wins. 


## Structure code base

The simulation can be used by running main.py. 

The program uses multiple objects to create the simulation. These are **Team**, **Aircraft**, **Target**, **Ground** and **Bullet**.

The classes **Target**, **Bullet** and **Ground** are used to create the environment. **Aircraft** is the superclass of the agent. **Aircraft** contains all physic related code allowing the plane to fly realistically. The subclass of **Aircraft**; **Agent**, contains the algorithms allowing the agent to function as an individual entity. 

The simulation can be run with or without a GUI. By setting the batch size in settings, the simulation can be ran *X* number of times. 

All global variables, such as plane size, are defined in settings.py. 

