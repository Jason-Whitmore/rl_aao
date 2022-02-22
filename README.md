# Reinforcement Learning: All Actions Optimization

This reinforcement learning (RL) project seeked to implement, test, and compare a new, novel RL algorithm to a traditional
RL algorithm similar to One Step TD learning.

The complete paper is available [here]()

## Motivation

Existing deep reinforcement learning (DRL) methods can be costly to train agents, both in regards to money and time.
Even training an agent in a simple environment like Atari Breakout can take millions of timesteps to achieve humanlike performance.
Due to these costs, it remains an open problem within DRL and finding more efficient methods of training is highly desirable.

## All Actions Optimization


## Experiment

In order to determine the performance of the AAO algorithm, an experiment will be conducted to determine
learning efficiency compared to a more traditional algorithm.

### Baseline algorithm: SAO


### Environments

The experiment will test both the AAO and SAO algorithms on 3 environments: CartPole-v1, LunarLander-v2, and Gridworld-10x10-1r-v0.
These environments are simple enough that a few runs can be conducted feasibly and vary enough to determine the algorithm's general learning abilities.

CartPole-v1 is a simple environment that tasks the agent with manuevering a car along an axis such that a vertical pole is balanced on top of it.
This environment is often used to test DRL algorithm implementations and can be learned reasonably fast, making it a good starting point for this experiment.

LunarLander-v2 is a more complex environment that tasks the agent with landing a spacecraft onto a lunar surface. The agent has access to a set of thrusters
to fire and seeks to land the spacecraft safely in a designated area. This environment provides a slightly more difficult challenge to a DRL algorithm compared
to the CartPole-v1 environment.

Gridworld-10x10-1r-v0 is a simple environment that tasks the agent with navigating through a 10 by 10 grid to collect a single reward, which once collected, ends
the environment. Unlike the other environments, this environment contains a sparse reward function, which gives the agent 0 reward on all actions except for when
the sole reward is collected, when a reward of 1 is given. This sparse reward function provides another challenge for both the AAO and SAO algorithm in the
experiment.

CartPole-v1 and LunarLander-v2 are environments provided by OpenAI's Gym package. The Gridworld-10x10-1r-v0 is a custom environment that was made for this project,
and is implemented using the same Gym interface that the other two environments use.

### Learning efficiency



### Experimental setup





## Results and Discussion


### CartPole-v1


### LunarLander-v2


### Gridworld-10x10-1r-v0


## How to run