#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py."""

from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np
from optimizer import BaseOptimizer


class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """


def my_matmul(x1, x2):
    """
    Given matrices x1 and x2, return the multiplication of them
    """
    result = np.zeros((x1.shape[0], x2.shape[1]))
    x1_non_zero_indices = x1.nonzero()
    if x1.shape[0] == 1 and len(x1_non_zero_indices[1]) == 1:
        result = x2[x1_non_zero_indices[1], :]
    elif x1.shape[1] == 1 and len(x1_non_zero_indices[0]) == 1:
        result[x1_non_zero_indices[0], :] = x2 * x1[x1_non_zero_indices[0], 0]
    else:
        result = np.matmul(x1, x2)
    return result


def get_value(s, weights):
    """
    Compute value of input s given the weights of a neural network
    """
    # Compute the output of the neural network, v, for input s
    num_states = s.shape[1]
    num_hidden_units = weights[0]["W"].shape[1]

    assert s.ndim == 2 and s.shape[0] == 1
    assert num_states == weights[0]["W"].shape[0]
    assert num_hidden_units == weights[1]["W"].shape[0]

    psi = my_matmul(s, weights[0]["W"]) + weights[0]["b"]
    assert psi.ndim == 2 and psi.shape == (1, num_hidden_units)

    x = np.maximum(0, psi)
    assert x.ndim == 2 and x.shape == (1, num_hidden_units)

    v = my_matmul(x, weights[1]["W"]) + weights[1]["b"]

    return v


def get_gradient(s, weights):
    """
    Given inputs s and weights, return the gradient of v with respect to the weights
    """
    grads = [dict() for i in range(len(weights))]

    psi = my_matmul(s, weights[0]["W"]) + weights[0]["b"]
    x = np.maximum(0, psi)

    # Compute the indicator function for ReLU gradient: I_x > 0
    I_x_gt_0 = (x > 0).astype(float)

    # Compute gradients according to the equations
    grads[0]["W"] = my_matmul(s.T, (weights[1]["W"].T * I_x_gt_0))
    grads[0]["b"] = weights[1]["W"].T * I_x_gt_0
    grads[1]["W"] = x.T
    grads[1]["b"] = np.array([[1.0]])

    return grads


def one_hot(state, num_states):
    """
    Given num_state and a state, return the one-hot encoding of the state
    """
    # Create the one-hot encoding of state
    # one_hot_vector is a numpy array of shape (1, num_states)
    one_hot_vector = np.zeros((1, num_states))
    one_hot_vector[0, int((state - 1))] = 1
    return one_hot_vector


class SGD(BaseOptimizer):
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """Setup for the optimizer.

        Set parameters needed to setup the stochastic gradient descent method.

        Assume optimizer_info dict contains:
        {
            step_size: float
        }
        """
        self.step_size = optimizer_info.get("step_size")

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                weights[i][param] += self.step_size * g[i][param]

        return weights


class Adam(BaseOptimizer):
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """Setup for the optimizer.

        Set parameters needed to setup the Adam algorithm.

        Assume optimizer_info dict contains:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float,
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
        }
        """

        self.num_states = optimizer_info.get("num_states")
        self.num_hidden_layer = optimizer_info.get("num_hidden_layer")
        self.num_hidden_units = optimizer_info.get("num_hidden_units")

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(self.num_hidden_layer + 1)]
        self.v = [dict() for i in range(self.num_hidden_layer + 1)]

        for i in range(self.num_hidden_layer + 1):
            # Initialize self.m[i]["W"], self.m[i]["b"], self.v[i]["W"], self.v[i]["b"] to zero
            self.m[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i + 1]))
            self.m[i]["b"] = np.zeros((1, self.layer_size[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i + 1]))
            self.v[i]["b"] = np.zeros((1, self.layer_size[i + 1]))

        # Initialize beta_m_product and beta_v_product to be later used for computing m_hat and v_hat
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """

        for i in range(len(weights)):
            for param in weights[i].keys():
                # update self.m and self.v
                self.m[i][param] = (
                    self.beta_m * self.m[i][param] + (1 - self.beta_m) * g[i][param]
                )
                self.v[i][param] = self.beta_v * self.v[i][param] + (
                    1 - self.beta_v
                ) * (g[i][param] * g[i][param])

                # compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                # update weights
                weights[i][param] += (
                    self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
                )

        # update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights


class TDAgent(BaseAgent):
    def __init__(self):
        self.name = "td_agent"
        pass

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD with a Neural Network.

        Assume agent_info dict contains:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float,
            discount_factor: float,
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
            seed: int
        }
        """

        # Set random seed for weights initialization for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Set attributes according to agent_info
        self.num_states = agent_info.get("num_states")
        self.num_hidden_layer = agent_info.get("num_hidden_layer")
        self.num_hidden_units = agent_info.get("num_hidden_units")
        self.discount_factor = agent_info.get("discount_factor")

        # Define the neural network's structure
        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # Initialize the neural network's parameter
        self.weights = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            # Initialize self.weights[i]["W"] and self.weights[i]["b"] using self.rand_generator.normal()
            self.weights[i]["W"] = self.rand_generator.normal(
                0,
                np.sqrt(2 / self.layer_size[i]),
                (self.layer_size[i], self.layer_size[i + 1]),
            )
            self.weights[i]["b"] = self.rand_generator.normal(
                0, np.sqrt(2 / self.layer_size[i]), (1, self.layer_size[i + 1])
            )

        # Specify the optimizer
        self.optimizer = Adam()
        self.optimizer.optimizer_init(
            {
                "num_states": agent_info["num_states"],
                "num_hidden_layer": agent_info["num_hidden_layer"],
                "num_hidden_units": agent_info["num_hidden_units"],
                "step_size": agent_info["step_size"],
                "beta_m": agent_info["beta_m"],
                "beta_v": agent_info["beta_v"],
                "epsilon": agent_info["epsilon"],
            }
        )

        self.last_state = None
        self.last_action = None

    def agent_policy(self, state):
        # Set chosen_action as 0 or 1 with equal probability.
        chosen_action = self.policy_rand_generator.choice([0, 1])
        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        # select action given state (using self.agent_policy()), and save current state and action
        self.last_state = state
        self.last_action = self.agent_policy(state=state)

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        # Compute TD error
        current_value = get_value(one_hot(state, self.num_states), self.weights)
        last_value = get_value(one_hot(self.last_state, self.num_states), self.weights)
        delta = reward + self.discount_factor * current_value - last_value

        # Retrieve gradients
        grads = get_gradient(
            s=one_hot(self.last_state, self.num_states), weights=self.weights
        )

        # Compute g
        g = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            for param in self.weights[i].keys():
                g[i][param] = delta * grads[i][param]

        # update the weights using self.optimizer
        self.weights = self.optimizer.update_weights(self.weights, g)

        # update self.last_state and self.last_action
        self.last_state = state
        self.last_action = self.agent_policy(state)

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # compute TD error
        last_value = get_value(one_hot(self.last_state, self.num_states), self.weights)
        delta = reward - last_value

        # Retrieve gradients
        grads = get_gradient(
            s=one_hot(self.last_state, self.num_states), weights=self.weights
        )

        # Compute g
        g = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            for param in self.weights[i].keys():
                g[i][param] = delta * grads[i][param]

        # update the weights using self.optimizer
        self.weights = self.optimizer.update_weights(self.weights, g)

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        if message == "get state value":
            state_value = np.zeros(self.num_states)
            for state in range(1, self.num_states + 1):
                s = one_hot(state, self.num_states)
                state_value[state - 1] = get_value(s, self.weights)
            return state_value
