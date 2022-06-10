from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import time
import gym
from csv_writer import csv_writer
from gridworld_10x10_1r_v0 import Gridworld10x10

class Experiment:
    """
    Defines the experiment class, which provides all of the functionality to conduct experiments comparing the AAO and SAO algorithms.
    """

    def __init__(self, env_name: str, hidden_layer_sizes: int, frame_stack: int):
        """
        Constructs the experiment object.

        env_name: The name of the OpenAI Gym environment that will be used in the experiment

        hidden_layer_sizes: The number of units in each hidden layer for the policy, value, and q function neural networks.

        frame_stack: The number of the most recent observations used when calculating the state vector.
        """

        #Initialize environment parameters
        self.environment_name: str = env_name #"CartPole-v1" #MsPacman-ramNoFrameskip-v4 "LunarLander-v2"
        self.obs_size: int = 0
        self.obs_lower_bound: list = []
        self.obs_upper_bound: list = []
        self.action_count: int = 0
        self.discount_factor: float = 0.999
        self.frame_stack: int = frame_stack
        self.obs_scalar: float = 1

        #Register the custom environment
        gym.register("Gridworld-10x10-1r-v0", entry_point=Gridworld10x10)

        #Model hyperparameters
        self.hidden_units = hidden_layer_sizes

        self.initialize_gym_env_variables(self.environment_name)

        #Policy fields
        self.untrained_policy_params = None
        self.action_prob_gradient_objects = []
        self.policy_function = self.create_policy_function()


        #Q function fields
        self.untrained_q_params = None
        self.q_gradient_object = None
        self.q_function = self.create_q_function()

        #Value function fields
        self.untrained_value_params = None
        self.value_gradient_object = None
        self.value_function = self.create_value_function()

        #Gradient objects created, create session object
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        

        self.obs_queue = []

    def change_environment(self, new_env_name: str):
        """
        Updates the parameters of the agent based on a new environment.

        new_env_name: The new environment name to change the parameters to.
        """

        self.environment_name = new_env_name

        self.initialize_gym_env_variables(self.environment_name)

    def reset_agent_episode(self):
        """
        Resets the internal observation queue so that a new episode can be started properly.
        """

        self.obs_queue = []

    def reset_agent_model_params(self):
        """
        Resets the neural network parameters to a pre-trained parameter set.
        """

        self.policy_function.set_weights(self.copy_list(self.untrained_policy_params))
        self.q_function.set_weights(self.copy_list(self.untrained_q_params))
        self.value_function.set_weights(self.copy_list(self.untrained_value_params))


    def get_state(self, obs: "ndarray") -> "ndarray":
        """
        Gets the state given an interal queue of observations and the most recent observation.

        obs: The most recent observation which will be added to the observation queue before the state vector is calculated.

        Returns the state vector.
        """

        #Process observation
        obs = self.process_observation(obs)

        state = []
        #Collect state from a series of 3 of the most recent observations

        if len(self.obs_queue) >= self.frame_stack:
            self.obs_queue.pop(len(self.obs_queue) - 1)
        self.obs_queue.insert(0, obs)
        
        state_length: int = self.obs_size * self.frame_stack

        #Populate the state vector
        for i in range(len(self.obs_queue)):
            for t in range(len(self.obs_queue[i])):
                value = self.obs_queue[i][t]
                state.append(value)
        
        #Fill remaining spots with invalid entries
        while len(state) < state_length:
            state.append(-1.0)

        return np.array(state, dtype="float32")


    def get_action(self, state: "ndarray") -> "Tuple[int, float]":
        """
        Selects an action from the current policy.

        state: The state vector which is used as input to the policy function.

        Returns a tuple of (action, probability of action)
        """

        softmax = self.policy_function.predict(np.array([state]))[0].tolist()

        n = np.random.random()

        index = 0
        for i in range(len(softmax)):
            if n < softmax[i]:
                index = i
                break
            else:
                n -= softmax[i]
        

        return index, softmax[index]
    

    def evaluate_policy_value_function(self, num_episodes: int) -> "list[float]":
        """
        Evaluates the current policy using the policy function and the q function to calculate the value function.

        num_episodes: The number of episodes to collect when evaluating the policy. A larger number means a more accurate average, but a slower time to calculate.

        Returns a list of scalar returns, one for each sample.
        """

        old_obs_queue = self.obs_queue
        scores = []

        env = gym.make(self.environment_name)

        for i in range(num_episodes):
            
            self.obs_queue = []
            obs = env.reset()
            state = self.get_state(obs)

            scores.append(self.value_function_predict(state))

            prob_vector: "ndarray" = self.policy_function_predict(state)
            q_vector: "ndarray" = self.q_function_predict_vector(state)

            v: float = 0
            for action in range(self.action_count):
                v += prob_vector[action] * q_vector[action]

            #scores.append(v)

        self.obs_queue = old_obs_queue
        return scores

                
    def evaluate_policy_monte_carlo(self, num_episodes: int, use_discount_factor: bool = True) -> "list[float]":
        """
        Evaluates the current policy using the monte carlo technique, which simply runs the policy in the environment, records rewards, and then averages them.

        num_episodes: The number of episodes to collect when evaluating the policy. A larger number means a more accurate average, but a slower time to calculate.

        use_discount_factor: If true, the result will be an approximation of the value function at the start state. If false, the result will be an approximation
        of the score, which does not use discounting (does not multiply future rewards by gamma^t).

        Returns a list of scalar returns, one for each sample.
        """

        old_obs_queue = self.obs_queue

        r = []

        discount_factor: float = 1

        if use_discount_factor:
            discount_factor = self.discount_factor

        env = gym.make(self.environment_name)

        while len(r) < num_episodes:
            

            obs = env.reset()
            done = False

            self.obs_queue = []
            state = self.get_state(obs)

            score = 0

            timestep = 0

            while not done:
                a, prob = self.get_action(state)

                obs, reward, done, info = env.step(a)
                score += (discount_factor ** timestep) * reward
                timestep += 1

                state = self.get_state(obs)

            r.append(score)
        
        self.obs_queue = old_obs_queue

        return r



            
    def initialize_gym_env_variables(self, env_name: str):
        """
        Populates various class fields based on the properties of the specified OpenAI Gym environment.

        env_name: The name of the OpenAI Gym environment.
        """
        
        env = gym.make(env_name)
        
        self.obs_lower_bound = env.observation_space.low
        self.obs_upper_bound = env.observation_space.high
        self.action_count = env.action_space.n

        


        self.obs_scalar = 1.0

        env.reset()
        
        obs, reward, done, info = env.step(env.action_space.sample())
        obs = self.process_observation(obs)


        self.obs_size = len(obs)
        
        env.reset()



    def q_function_predict_vector(self, state: "ndarray") -> "ndarray":
        """
        Makes a prediction using the q function neural network.

        state: The state input vector to q function neural network.

        Returns a numpy array with each component corresponding to the q value of each action at the specified state.
        """

        state_action_vectors: "list[ndarray]" = []

        for a in range(self.action_count):
            state_action_vectors.append(self.get_state_action_vector(state, a))

        vector: "ndarray" = self.q_function.predict_on_batch(np.array(state_action_vectors)).flatten()

        return vector


    def policy_function_predict(self, state: "ndarray") -> "ndarray":
        """
        Makes a prediction using the policy function neural network.

        state: The state input vector to the value function neural network.

        Returns the scalar output of the value function neural network.
        """

        return self.policy_function.predict(np.array([state]))[0].flatten()

    def value_function_predict(self, state: "ndarray") -> float:
        """
        Makes a prediction using the value function neural network.

        state: The state input vector to the value function neural network.

        Returns the scalar output of the value function neural network.
        """

        return self.value_function.predict(np.array([state])).flatten()[0]


    def get_state_action_vector(self, state_vector: "ndarray", action: int) -> "ndarray":
        """
        Creates a state action vector which is used as the input vector for the q function neural network. This works by simply concatenating the state vector
        and a one-hot encoded action vector.

        state_vector: The state vector that will be used in concatenation.

        action: The action integer which will be used to make a one-hot encoded representation, which will then be concatenated with the state_vector

        Returns the concatenated state_action vector.
        """
        r = np.zeros((len(state_vector) + self.action_count))

        for i in range(len(state_vector)):
            r[i] = state_vector[i]

        r[len(state_vector) + action] = 1

        return r


    def create_policy_function(self):
        """
        Creates the policy function neural network to be used for both the SAO and AAO algorithms. Hyperparameters are defined in the class fields.
        This function also sets up the gradient objects, which are used to adjust the parameters without calling a fit() function.

        Returns the constructed policy function neural network.
        """

        input_size: int = self.obs_size * self.frame_stack
                        
        policy = Sequential()
        policy.add(Dense(self.hidden_units, input_dim=input_size, activation="tanh"))
        policy.add(Dense(self.hidden_units, activation="tanh"))

        policy.add(Dense(self.action_count, activation="softmax", kernel_initializer="zero"))
        policy.compile(optimizer="sgd", loss="mse")

        self.untrained_policy_params = self.copy_list(policy.get_weights())

        #Set up gradient objects
        for a in range(self.action_count):
            self.action_prob_gradient_objects.append(tf.gradients(policy.output[:, a], policy.trainable_weights))


        return policy

    def create_value_function(self):
        """
        Creates the value function neural network to be used for the SAO algorithm. Hyperparameters are defined in the class fields.
        This function also sets up the gradient objects, which are used to adjust the parameters without calling a fit() function.

        Returns the constructed value function neural network.
        """


        #Create model
        input_size: int = self.obs_size * self.frame_stack

        value_function = Sequential()
        value_function.add(Dense(self.hidden_units, input_dim=input_size, activation="tanh"))
        value_function.add(Dense(self.hidden_units, activation="tanh"))

        value_function.add(Dense(1, activation="linear"))

        value_function.compile(optimizer="sgd", loss="mse")

        self.untrained_value_params = self.copy_list(value_function.get_weights())

        #Set up gradient object
        self.value_gradient_object = tf.gradients(value_function.output[:, 0], value_function.trainable_weights)

        return value_function



    def create_q_function(self):
        """
        Creates the q function neural network to be used for the AAO algorithm. Hyperparameters are defined in the class fields.
        This function also sets up the gradient objects, which are used to adjust the parameters without calling a fit() function.

        Returns the constructed q function neural network.
        """

        input_size: int = (self.obs_size * self.frame_stack) + self.action_count

        q = Sequential()
        q.add(Dense(self.hidden_units, input_dim=input_size, activation="tanh"))
        q.add(Dense(self.hidden_units, activation="tanh"))
        q.add(Dense(1, activation="linear"))

        q.compile(optimizer="sgd", loss="mse")

        #Save untrained params
        self.untrained_q_params = self.copy_list(q.get_weights())

        #Create gradient object
        self.q_gradient_object = tf.gradients(q.output[:, 0], q.trainable_weights)

        return q


    def watch_game(self, delay: float = 1.0/20):
        """
        Loads the saved policy parameters at policy.h5, then runs the policy and renders the environment so that the user can see a trained agent in action.
        This function is mostly used for debug purposes.

        delay: The time, in seconds, that should pass before performing the next timestep.
        """
        
        self.policy_function.load_weights("policy.h5")

        env = gym.make(self.environment_name)
        timestep: int = 0
            

        obs = env.reset()
        done = False

        self.obs_queue = []
        state = self.get_state(obs)

        score = 0

        while not done:
            a, prob = self.get_action(state)
            timestep += 1

            env.render()
            obs, reward, done, info = env.step(a)
            
            time.sleep(delay)

            score += reward

            state = self.get_state(obs)

        print("Timesteps: " + str(timestep))

        exit()

    def process_observation(self, obs: "ndarray") -> "ndarray":
        """
        Performs adjustments to an observation vector in order to make the components ready to be used by a neural network as input.
        These adjustments are dependent on the environment and format of the observation. For example, atari ram environments need observation
        vectors to be converted to raw unstructured bits in order to be used.

        obs: The observation vector, obtained directly from the OpenAI Gym API.

        Returns a processed observation that is ready to be used in a state input for a neural network.
        """

        if "ram" in self.environment_name:
            #Obs will be an array of size 128. Convert to 1024 bits
            new_obs = []
            for i in range(len(obs)):
                new_obs.extend(self.int_to_binary_list(obs[i]))
            return new_obs
        else:
            return obs * self.obs_scalar


    def int_to_binary_list(self, n: int) -> "list[int]":
        """
        Converts the input integer into the binary representation as a list of integers, containing either 1 or 0.

        n: The input integer to convert to binary.

        Returns a list of binary digits representing the input integer value.
        """

        r = []

        bin_string = bin(n)
        bin_string = bin_string[2:]

        while len(bin_string) < 8:
            bin_string = "0" + bin_string

        for i in range(len(bin_string)):
            if bin_string[i] == "1":
                r.append(1)
            else:
                r.append(0)


        return r

    def copy_list(self, l: "list[ndarray]") -> "list[ndarray]":
        """
        Makes a copy of the input list of numpy arrays.

        l: The list of numpy arrays to copy

        Returns the copy of l.
        """
        r = []

        for i in range(len(l)):
            r.append(np.copy(l[i]))

        return r
    
    def clear_list(self, l: "list[ndarray]") -> "list[ndarray]":
        """
        Clears all entries in the list of numpy arrays.

        l: The list of numpy arrays to clear.

        Returns the cleared l list.
        """

        for i in range(len(l)):
            l[i] = np.zeros(l[i].shape)

        return l

    def scale_list(self, l: "list[ndarray]", scalar: float) -> "list[ndarray]":
        """
        Scales the components of the given list of numpy arrays by scalar.

        l: The list of numpy arrays to scale elementwise.

        scalar: The scalar value to multiply each component in l by.

        Returns the scaled list, the modified l list.
        """
        for i in range(len(l)):
            l[i] *= scalar

        return l

    def add_list(self, l: "list[ndarray]", dest: "list[ndarray]") -> "list[ndarray]":
        """
        Adds components of 2 list of numpy arrays elementwise.

        l: The first list

        dest: The second list. This is the list that is modified

        Returns the dest list, where the result of the addition is placed.
        """

        for i in range(len(l)):
            dest[i] += l[i]

        return dest

    

    def q_gradient(self, state: "ndarray", action: int) -> "list[ndarray]":
        """
        Calculates the gradient of the q function with a given action and state.

        state: The state vector input, which is used to calculate the gradient

        action: The action as part of the (s,a) pair that is used as input to the q function.

        Returns the gradient as a list of numpy arrays.
        """

        state_action: "ndarray" = self.get_state_action_vector(state, action)

        state_action = np.reshape(state_action, (1, len(state_action)))

        return self.session.run(self.q_gradient_object, feed_dict={self.q_function.input : state_action})



    def policy_gradient(self, state: "ndarray", action: int) -> "list[ndarray]":
        """
        Calculates the gradient of the policy function probability with a given action and state.

        state: The state vector input, which is used to calculate the gradient

        action: The action, or action index, which specifies which part of the output vector the gradient will be calculated for

        Returns the gradient as a list of numpy arrays.
        """

        state = np.array([state])

        return self.session.run((self.action_prob_gradient_objects[action]), feed_dict={self.policy_function.input : state})
        
    def value_gradient(self, state: "ndarray") -> "list[ndarray]":
        """
        Calculates the gradient of the value function output given a specific state input.

        state: The state vector input, which is used to calculate the gradient

        Returns the gradient as a list of numpy arrays.
        """

        state: "ndarray" = np.array([state])

        return self.session.run(self.value_gradient_object, feed_dict={self.value_function.input: state})

    def td_aao(self, num_timesteps: int, score_limit: float, policy_lr: float, q_lr: float, results_filepath: str=None, evaluation_interval: int=1000 * 1000, eval_num_trials: int=500):
        """
        Trains a policy using the all action optimization algorithm. Will display statistics to standard output, save trained policies,
        and write results to disk

        num_timesteps: The total number of timesteps to run the SAO training algorithm for. This does not include timesteps used for evaluating the policy

        score_limit: The experiment stops when the evaluated mean score meets or exceeds this value.

        policy_lr: The learning rate for the policy function, should be > 0.

        q_lr: The learning rate for the q function, should be > 0.

        evaluation_interval: The interval, in timesteps, between policy evaluations

        num_samples: The number of samples (episodes) used to calculate the mean score at every evaluation interval.

        results_filepath: The filepath to save the results csv file to. 
        """

        #Gym startup
        env = gym.make(self.environment_name)

        #Create observation queue to keep track of observations
        obs_queue = []

        total_timesteps: int = 0

        avg_q_delta: float = 0.0
        avg_action_prob: float = 0.0

        #Create the writer object
        writer: csv_writer = csv_writer(results_filepath, ["Timestep", "Mean score"])

        while total_timesteps < num_timesteps:
            #Start of episode

            obs = env.reset()
            self.reset_agent_episode()

            #Retrieve the current state
            state = self.get_state(obs)
            done = False

            episode_timesteps = 0
            start_value: float = 0

            while not done and total_timesteps < num_timesteps:
                #Select action
                a, prob = self.get_action(state)

                #Perform action, get next observation, reward
                next_obs, r, done, info = env.step(a)
                
                total_timesteps += 1
                episode_timesteps += 1

                #Get next state
                next_state: "ndarray" = self.get_state(next_obs)

                #Policy evaluation
                v_prime: float = 0

                q_vector: "ndarray" = self.q_function_predict_vector(state)
                q_prime_vector: "ndarray" = self.q_function_predict_vector(next_state)

                prob_vector: "ndarray" = self.policy_function_predict(state)
                prob_prime_vector: "ndarray" = self.policy_function_predict(next_state)

                

                if not done:
                    for action in range(self.action_count):
                        v_prime += q_prime_vector[action] * prob_prime_vector[action]
                
                q_hat: float = r + (self.discount_factor * v_prime)


                delta_q: float = q_hat - q_vector[a]
                

                avg_q_delta += 0.0001 * (delta_q - avg_q_delta)
                avg_action_prob += 0.0001 * (prob_vector[a] - avg_action_prob)
                

                #Adjust q function
                q_params = self.copy_list(self.q_function.get_weights())
                q_gradient = self.q_gradient(state, a)
                q_gradient_scaled = self.scale_list(q_gradient, delta_q * q_lr)

                new_q_params = self.add_list(q_gradient_scaled, q_params)

                self.q_function.set_weights(new_q_params)

                #Policy improvement
                policy_params: list["ndarray"] = self.copy_list(self.policy_function.get_weights())
                g: list["ndarray"] = self.copy_list(self.policy_function.get_weights())
                g = self.clear_list(g)

                q_vector: "ndarray" = self.q_function_predict_vector(state)

                #Recalculate V(s)
                v: float = 0.0
                for action in range(self.action_count):
                    v += q_vector[action] * prob_vector[action]


                for action in range(self.action_count):
                    delta_a: float = q_vector[action] - v

                    #Get gradient
                    grad_a: list["ndarray"] = self.policy_gradient(state, action)

                    #Add gradient to g scaling by delta
                    grad_a_scaled: list["ndarray"] = self.scale_list(grad_a, delta_a)
                    g = self.add_list(grad_a_scaled, g)

                
                #Add g to parameters of policy
                g_scaled = self.scale_list(g, policy_lr)
                new_policy_params: list["ndarray"] = self.add_list(g_scaled, policy_params)
                self.policy_function.set_weights(new_policy_params)
                
                state = next_state
                


                #Check if at logging interval
                if evaluation_interval != -1 and total_timesteps % evaluation_interval == 0:

                    avg_score_undiscounted: float = np.mean(self.evaluate_policy_monte_carlo(eval_num_trials, use_discount_factor=False))

                    print("Timestep: " + str(total_timesteps))
                    print("Avg score: " + str(avg_score_undiscounted))
                    self.policy_function.save_weights("policy.h5")
                    print("Avg q delta: " + str(abs(avg_q_delta)))
                    print("Avg action prob: " + str(avg_action_prob))
                    print()

                    #Write the results to file
                    writer.add_row([str(total_timesteps), str(avg_score_undiscounted)])
                    writer.write_to_file()

                    if avg_score_undiscounted >= score_limit:
                        print("Score limit achieved. Training stopped.")
                        return

                    
        

    def td_sao(self, num_timesteps: int, score_limit: float, policy_lr: float, v_lr: float, results_filepath: str=None, evaluation_interval: int=1000 * 1000, eval_num_trials: int=500):
        """
        Trains a policy using the  single action optimization algorithm. Will display statistics to standard output, save trained policies,
        and write results to disk

        num_timesteps: The total number of timesteps to run the SAO training algorithm for. This does not include timesteps used for evaluating the policy

        score_limit: The experiment stops when the evaluated mean score meets or exceeds this value.

        policy_lr: The learning rate for the policy function, should be > 0.

        v_lr: The learning rate for the value function, should be > 0.

        evaluation_interval: The interval, in timesteps, between policy evaluations

        num_samples: The number of samples (episodes) used to calculate the mean score at every evaluation interval.

        results_filepath: The filepath to save the results csv file to. 
        """
        
        #Gym startup
        env = gym.make(self.environment_name)

        total_timesteps: int = 0

        avg_td: float = 0
        avg_prob: float = 0

        #Create the writer object
        writer: csv_writer = csv_writer(results_filepath, ["Timestep", "Mean score"])

        while total_timesteps < num_timesteps:
            #Start of episode

            obs = env.reset()
            self.reset_agent_episode()

            #Retrieve the current state
            state = self.get_state(obs)
            done = False

            episode_timesteps = 0
            start_value: float = 0

            while not done and (total_timesteps < num_timesteps):
                #Select action
                a, prob = self.get_action(state)

                avg_prob += 0.001 * (prob - avg_prob)

                #Perform action, get next observation, reward
                next_obs, r, done, info = env.step(a)
                
                total_timesteps += 1
                episode_timesteps += 1

                #Get next state
                next_state: "ndarray" = self.get_state(next_obs)

                #Get TD error
                v: float = self.value_function_predict(state)

                v_prime: float = 0

                if not done:
                    v_prime = self.value_function_predict(next_state)
                
                delta: float = (r + self.discount_factor * v_prime) - v

                avg_td += 0.001 * (delta - avg_td)
                

                #Adjust value function
                value_params = self.copy_list(self.value_function.get_weights())
                value_gradient = self.value_gradient(state)
                value_gradient_scaled = self.scale_list(value_gradient, delta * v_lr)

                new_value_params = self.add_list(value_gradient_scaled, value_params)

                self.value_function.set_weights(new_value_params)

                #Policy improvement
                policy_params: list["ndarray"] = self.copy_list(self.policy_function.get_weights())
                policy_gradient: "list[ndarray]" = self.policy_gradient(state, a)
                gradient_scaled = self.scale_list(policy_gradient, delta * policy_lr * 1)
                new_policy_params: list["ndarray"] = self.add_list(gradient_scaled, policy_params)
                self.policy_function.set_weights(new_policy_params)
                
                state = next_state
                


                #Check if at logging interval
                if evaluation_interval != -1 and total_timesteps % evaluation_interval == 0:
                    scores = self.evaluate_policy_monte_carlo(30, use_discount_factor=True)
                    avg_score_monte_carlo = np.mean(scores)
                    avg_score_undiscounted: float = np.mean(self.evaluate_policy_monte_carlo(eval_num_trials, use_discount_factor=False))

                    print("Timestep: " + str(total_timesteps))
                    print("Avg score: " + str(avg_score_undiscounted))
                    print("Avg start value (monte carlo): " + str(avg_score_monte_carlo))
                    #print("Avg start value (value function): " + str(np.mean(self.evaluate_policy_value_function(10))))
                    self.policy_function.save_weights("policy.h5")
                    print("Avg td: " + str(avg_td))
                    print("Avg prob: " + str(avg_prob))
                    print()

                    #Write the results to file
                    writer.add_row([str(total_timesteps), str(avg_score_undiscounted)])
                    writer.write_to_file()

                    if avg_score_undiscounted >= score_limit:
                        print("Score limit achieved. Training stopped.")
                        return
            

    def run_score_over_time_sao(self, num_timesteps: int, score_limit: float, policy_lr: float, v_lr: float, evaluation_interval: int, num_samples: int, results_filepath: str=None):
        """
        Runs the score over time experiment for the single action optimization algorithm. Will display statistics to standard output, save trained policies,
        and write results to disk

        num_timesteps: The total number of timesteps to run the SAO training algorithm for. This does not include timesteps used for evaluating the policy

        score_limit: The experiment stops when the evaluated mean score meets or exceeds this value.

        policy_lr: The learning rate for the policy function, should be > 0.

        v_lr: The learning rate for the value function, should be > 0.

        evaluation_interval: The interval, in timesteps, between policy evaluations

        num_samples: The number of samples (episodes) used to calculate the mean score at every evaluation interval.

        results_filepath: The filepath to save the results csv file to. 
        """
        
        print("Score over time experiment for single action optimization on environment: " + str(self.environment_name))


        self.td_sao(num_timesteps=num_timesteps, score_limit=score_limit, policy_lr=policy_lr, v_lr=v_lr, evaluation_interval=evaluation_interval, eval_num_trials=num_samples, results_filepath=results_filepath)


    def run_score_over_time_aao(self, num_timesteps: int, score_limit: float, policy_lr: float, q_lr: float, evaluation_interval: int, num_samples: int, results_filepath: str=None):
        """
        Runs the score over time experiment for the all action optimization algorithm. Will display statistics to standard output, save trained policies,
        and write results to disk

        num_timesteps: The total number of timesteps to run the SAO training algorithm for. This does not include timesteps used for evaluating the policy

        score_limit: The experiment stops when the evaluated mean score meets or exceeds this value.

        policy_lr: The learning rate for the policy function, should be > 0.

        q_lr: The learning rate for the q function, should be > 0.

        evaluation_interval: The interval, in timesteps, between policy evaluations

        num_samples: The number of samples (episodes) used to calculate the mean score at every evaluation interval.

        results_filepath: The filepath to save the results csv file to. 
        """
        
        print("Score over time experiment for all action optimization on environment: " + str(self.environment_name))


        self.td_aao(num_timesteps=num_timesteps, score_limit=score_limit, policy_lr=policy_lr, q_lr=q_lr, evaluation_interval=evaluation_interval, eval_num_trials=num_samples, results_filepath=results_filepath)

print("Numpy version: " + np.__version__ + ", should be equal to 1.14.5")
print("Tensorflow version: " + tf.__version__ + ", should be equal to 1.10.0")
print("Keras version: " + keras.__version__ + ", should be equal to 2.2.4")

index = 2

env_names = ["CartPole-v1", "LunarLander-v2", "Gridworld-10x10-1r-v0"] #MsPacman-ramDeterministic-v4
frame_stack = [2, 2, 1]
hidden_layer_sizes = [24, 32, 32]
score_targets = [490, 195, 0]

policy_lrs = [1e-6, 1e-5, 1e-4]
value_lrs = [1e-5, 1e-4, 1e-2]

#normally 2 * 1000 * 1000
timestep_limit: int = 2 * 1000 * 1000

#normally 10 * 1000
evaluation_interval: int = 10 * 1000

#normally 30
num_samples_evaluation: int = 30

result_filepath = "exp_results_" + env_names[index] + ".csv"


agent = Experiment(env_names[index], hidden_layer_sizes[index], frame_stack[index])

#agent.watch_game()
#exit()
#Determine mean random score
print("Mean random policy score for " + str(env_names[index]) + ": " + str(np.mean(agent.evaluate_policy_monte_carlo(num_samples_evaluation, use_discount_factor=False))))

#agent.run_score_over_time_aao(num_timesteps=timestep_limit, score_limit=score_targets[index], policy_lr=policy_lrs[index], q_lr=value_lrs[index], evaluation_interval=evaluation_interval, num_samples=num_samples_evaluation, results_filepath=result_filepath)
agent.run_score_over_time_sao(num_timesteps=timestep_limit, score_limit=score_targets[index], policy_lr=policy_lrs[index], v_lr=value_lrs[index], evaluation_interval=evaluation_interval, num_samples=num_samples_evaluation, results_filepath=result_filepath)