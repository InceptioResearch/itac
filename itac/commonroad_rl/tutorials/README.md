# Tutorials to CommonRoad-RL
This set of tutorials aims at smoothing your start with the [CommonRoad-RL](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl) package.  
In the following, **Theoretical Primer** lists out several recommended readings and guides to the background knowledge. Then, **Practical Exercises** consisting of several interactive python notebooks demonstrate the most essential functionalities in the [CommonRoad-RL](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl) package.

## Installation
* Please follow the [README.md](https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl/-/blob/development/README.md) to install the CommonRoad-RL package and all dependencies.
* Install the corresponding ipykernel for the tutorials.
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=nameOfYourCondaEnv
```
Make sure that you always choose the correct kernel in the tutorials.

## Theoretical Primer

### CommonRoad
Please refer to [CommonRoad](https://commonroad.in.tum.de) for detailed documentations and tutorials.

### Reinforcement Learning
The following links point to places that provide helpful RL introductions and software package guides. 

* [OpenAI Stable Baselines](https://stable-baselines.readthedocs.io/en/master/guide/quickstart.html) *(Mandatory)*  
This is a host of great documentations and implementations of various [RL algorithms](https://stable-baselines.readthedocs.io/en/master/guide/algos.html), on which CommonRoad-RL highly leverages. A walkthrough of essential functions can be found [here](https://stable-baselines.readthedocs.io/en/master/guide/examples.html). 

* [OpenAI Gym](https://gym.openai.com/docs/) *(Mandatory)*  
This library provides a collection of test problems, also called RL environments, such as Cart Pole control and Atari games. Additionally, it allows construction of customized environments through the convenient interface, thereby resulting in our `commonroad-v0` for CommonRoad motion planning problems.

* [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/#) *(Optional)*  
This serves as a nice and clear introduction to RL, showing the theoretical background and key concepts briefly. Nevertheless, CommonRoad-RL does not employ implementations from this package.

* [OpenAI Safety Gym](https://github.com/openai/safety-gym) *(Optional)*  
This package, although not directly imported in CommonRoad-RL, inspires us with its dictionary-based realization of RL environments. Such implementation enables not only easy configuration of RL environment components, but also convenient accesses to these items from other functions or files. 

## Practical Exercises
Before proceeding, please make sure you have obtained fundamental ideas of CommonRoad packages, RL environments, and RL algorithms, with the aforementioned links. Now, please follow the interactive python notebooks in the suggested order.

Note that we assume our working directory at the project root `commonroad-rl`, and that all paths in the exercises are relative paths to the working directory. Please adjust any variables if applicable.
