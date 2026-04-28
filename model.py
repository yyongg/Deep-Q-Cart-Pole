"""
MVC - Model

This module is the reinforcement learning environment inheriting from
Gymnasium's cartpole. It implements a reward function designed to train the agent

Reward penalizations
    1. Pole angle (tilt from vertical)
    2. Cart distance from the center

"""

import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv
import gymnasium as gym


class CustomCartPole(CartPoleEnv):  # pylint: disable=too-many-instance-attributes
    """
    This class uses the Gymnasium CartPole environment to allow
    the controller to set specific physical properties (length, mass) and defines
    a reward function

    Attributes:
        gravity (float): Acceleration due to gravity (default 9.8 m/s^2).
        length (float): Half-length of the pole (distance to center of mass).
        masspole (float): Mass of the pole in kilograms.

    """

    def __init__(self, render_mode=None):
        """Initializes the environment with RGB rendering enabled for the View."""
        super().__init__(render_mode=render_mode)
        self.gravity = 9.81
        self.length = 0.4
        self.masspole = 0.1
        self.max_episode_steps = 500

        high = np.array(
            [
                4.8,
                np.finfo(np.float32).max,  # pylint: disable=no-member
                1.0,
                1.0,
                np.finfo(np.float32).max,  # pylint: disable=no-member
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def set_parameters(self, length, weight):
        """
            Overides the length and weight with the given user input.

        Args:
            length: An integer that represents the length of the pole in meters

            weight: An integer that represents the mass of the pole in kg

            Returns:
                None - sets class attributes of self.length and self.masspole to
                length and weight respectively
        """
        self.length = length
        self.masspole = weight

        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    def reset(self, *, seed=None, options=None):
        """Reset the environment to a fixed upright starting state.

        Overrides the default random initialisation so that every episode
        begins with the cart and pole perfectly stationary at the origin.

        Note the '*' enforces keyword-only arguments to match the
        Gymnasium base class signature.

        Args:
            seed (int | None): Random seed forwarded to the parent reset.
                Has no practical effect here because the state is hard-coded,
                but is accepted for API compatibility.
            options (dict | None): Currently unused; accepted for API
                compatibility with the Gymnasium interface.

        Returns:
            tuple[np.ndarray, dict]: A tuple of:
                - obs (np.ndarray): Initial observation ``[cart_pos, cart_vel, cos(angle),
                  sin(angle), pole_vel]``
                  representing cart position, cart velocity, pole angle trig, and
                  pole angular velocity respectively.
                - info (dict): Empty info dictionary.
        """
        super().reset(seed=seed)
        self.state = np.array(
            [0, 0, np.pi + np.random.uniform(-0.1, 0.1), 0], dtype=np.float32
        )

        obs = np.array(
            [
                self.state[0],
                self.state[1],
                np.cos(self.state[2]),
                np.sin(self.state[2]),
                self.state[3],
            ],
            dtype=np.float32,
        )

        return obs, {}

    def step(self, action):
        """Advance the simulation by one timestep with a shaped reward.

        Applies the given action via the parent ``step``, then replaces the
        default binary (+1 / episode-end) reward with a dense signal that
        penalises pole tilt, cart displacement, cart speed, and momentum
        directed away from the centre.

        Args:
            action (int): Discrete action — ``0`` to push the cart left,
                ``1`` to push the cart right.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: A tuple of:
                - obs (np.ndarray): Current observation
                  ``[cart_pos, cart_vel, cos(angle), sin(angle), pole_vel]``.
                - reward (float): Shaped reward for swing-up and balancing;
                  a penalty of ``-5.0`` is applied on termination.
                - terminated (bool): ``True`` when the cart has left the track boundary.
                - truncated (bool): ``True`` when the episode step limit is reached.
                - info (dict): Auxiliary diagnostic information forwarded from
                  the parent environment.
        """
        next_obs_raw, _, _, truncated, info = super().step(action)

        cart_pos = next_obs_raw[0]
        cart_vel = next_obs_raw[1]
        pole_angle = next_obs_raw[2]
        pole_vel = next_obs_raw[3]

        obs = np.array(
            [cart_pos, cart_vel, np.cos(pole_angle), np.sin(pole_angle), pole_vel],
            dtype=np.float32,
        )

        normalized_angle = abs(((pole_angle + np.pi) % (2 * np.pi)) - np.pi)
        normalized_pos = abs(cart_pos) * np.pi / 2.4

        total = (
            ((np.cos(normalized_angle) + 1) / 2)
            * max(0.0, float(np.cos(normalized_pos)))
            * np.cos(min(np.pi, abs(pole_vel) / 10.0))
        )

        reward = float(total - 0.1)
        terminated = bool(abs(cart_pos) > 2.4)

        if terminated:
            reward -= 5.0
            terminated = True

        return obs, reward, terminated, truncated, info
