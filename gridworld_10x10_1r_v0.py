import gym
from gym import spaces
import numpy as np
import random

class Gridworld10x10(gym.Env):
    """
    Defines the 10x10 gridworld with 5 rewards by extending the OpenAI gym interface
    """

    def __init__(self):
        """
        Initializes the gridworld environment and creates data structures to keep track of agent
        and reward locations, as well as populating the action, reward, and observation spaces.
        """
        super().__init__()

        self.num_rewards: int = 1

        self.grid_size: int = 10

        self.agent_x: int = np.random.randint(0, self.grid_size)
        self.agent_y: int = np.random.randint(0, self.grid_size)

        self.timestep_limit: int = 10000

        self.action_space = spaces.Discrete(5)
        self.reward_range = (-1, 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.get_observation_size(),))

        self.current_timestep = 0
        self.reward_coordinates = self.generate_reward_coordinates(num_coordinates=self.num_rewards)
        self.reward_coordinates_remaining = set(self.reward_coordinates)


    def generate_reward_coordinates(self, num_coordinates: int) -> "set(x,y)":
        """
        Generates reward locations given the number of coordinates desired.
        These locations are unique and are generated in O(kn^2) time, where n is the length
        of one of the board sizes and k is the number of coordinates to generate.

        num_coordinates: The number of rewards to generate unique coordinates for.

        Returns a set of tuples, with each tuple specifying the x and y coordinates as integers in range [0, 9]
        """
        coord_set = set()

        #Generate all possible tuples
        all_possible_tuples = set()

        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                all_possible_tuples.add(tuple((x, y)))

        #Randomly remove tuples an add to coord set

        tuple_list = list(all_possible_tuples)

        np.random.shuffle(tuple_list)

        for i in range(0, num_coordinates):
            t = tuple_list[i]

            coord_set.add(t)

        return coord_set

    def get_observation(self) -> "ndarray":
        """
        Returns a numpy array of length 17, which contains coordinates and availability status of each of the rewards, as well as the agent location
        as coordinates.
        """
        obs: "ndarray" = np.zeros(self.get_observation_size(), dtype="float32")

        index: int = 0
        for (x, y) in self.reward_coordinates:
            obs[index] = float(x) / self.grid_size
            index += 1
            
            obs[index] = float(y) / self.grid_size
            index += 1

            if (x,y) in self.reward_coordinates_remaining:
                obs[index] = 1.0
            
            index += 1

        obs[index] = float(self.agent_x) / self.grid_size
        index += 1

        obs[index] = float(self.agent_y) / self.grid_size
        index += 1

        obs[index] = float(self.current_timestep) / self.timestep_limit

        return obs

    def get_observation_size(self) -> int:
        """
        Returns the size of the observation, which needs to include agent x, y location, number of timesteps,
        the location of rewards, and whether or not each reward is collected or not (as a binary value)
        """
        return 3 + (self.num_rewards * 3)

        
    def step(self, action: int) -> "obs, reward, episode done?, info (None)":
        """
        Performs an action within the environment.

        action: The action to be performed, this action must be in the set {0, 1, 2, 3, 4}

        Returns a tuple of information to be used by the reinforcement learning algorithm.
        The observation contains information about the current environment state, the reward
        serves as a learning signal, and the episode done boolean alerts the reinforcement learning
        algorithm of the episode being completed. A None object is returned in place of a info object.
        """
        
        #index:      [  0     1     2    3      4   ]   
        #actions are [left, right, up, down, acquire]

        new_agent_x: int = self.agent_x
        new_agent_y: int = self.agent_y

        reward: float = 0

        #Apply action
        if action == 0:
            new_agent_x -= 1
        elif action == 1:
            new_agent_x += 1
        elif action == 2:
            new_agent_y -= 1
        elif action == 3:
            new_agent_y += 1

        #Determine reward
        if action == 4 and (new_agent_x, new_agent_y) in self.reward_coordinates_remaining:
            #Acquired a reward
            reward = 1.0

            #Remove reward from coordinates
            self.reward_coordinates_remaining.remove((new_agent_x, new_agent_y))
        else:

            for x, y in self.reward_coordinates_remaining:
                reward -= (abs(self.agent_x - x) + abs(self.agent_y - y)) / 1000.0


            #reward += -0.001

        #Make sure agent does not "wander" off of the map
        new_agent_x = min(self.grid_size - 1, new_agent_x)
        new_agent_x = max(0, new_agent_x)
        self.agent_x = new_agent_x

        new_agent_y = min(self.grid_size - 1, new_agent_y)
        new_agent_y = max(0, new_agent_y)
        self.agent_y = new_agent_y


        #Update timestep
        self.current_timestep += 1

        #Determine if episode is ended
        done: boolean = (self.current_timestep >= self.timestep_limit) or (len(self.reward_coordinates_remaining) == 0)

        obs = self.get_observation()

        return obs, reward, done, None

    def reset(self):
        """
        Resets the environment by placing the agent at a random spot, creating new rewards, and resetting
        a timestep counter.

        Returns the starting observation
        """
        self.agent_x = np.random.randint(0, self.grid_size)
        self.agent_y = np.random.randint(0, self.grid_size)

        self.current_timestep = 0
        self.reward_coordinates = self.generate_reward_coordinates(num_coordinates=self.num_rewards)
        self.reward_coordinates_remaining = set(self.reward_coordinates)

        return self.get_observation()

    def render(self):
        """
        Prints information about the current environment state, which includes the agent coordinates and
        coordinates for remaining rewards.
        """
        
        print("Agent coords: (" + str(self.agent_x) + ", " + str(self.agent_y) + ")")
        print("Remaining reward locations: " + str(self.reward_coordinates_remaining))
        print()

