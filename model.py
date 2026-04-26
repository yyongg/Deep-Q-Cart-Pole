import numpy as np
import pygame
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control import CartPoleEnv
from stable_baselines3 import PPO

class CustomCartPole(CartPoleEnv):
    """Matches the physics used during training (gravity, pole length).
    No random disturbances here — those were only needed for training."""

    def __init__(self):
        super().__init__(render_mode="rgb_array")
        self.gravity = 9.8
        self.length  = 0.5
        self.masspole = 0 # default values

    def set_parameters(self,length,weight):
        """
            Overides the length and weight with the given user input.

        Args:
            length: An integer that represents the length of the pole in meters

            weight: An integer that represents the mass of the pole in kg

            Returns:
                None - sets class attributes of self.length and self.masspole to length and weight respectively
        """
        self.length = length
        self.masspole = weight



    def reset(self, seed=None, options=None):
        """Reset the environment to a fixed upright starting state.

        Overrides the default random initialisation so that every episode
        begins with the cart and pole perfectly stationary at the origin.

        Args:
            seed (int | None): Random seed forwarded to the parent reset.
                Has no practical effect here because the state is hard-coded,
                but is accepted for API compatibility.
            options (dict | None): Currently unused; accepted for API
                compatibility with the Gymnasium interface.

        Returns:
            tuple[np.ndarray, dict]: A tuple of:
                - obs (np.ndarray): Initial observation ``[0, 0, 0, 0]``
                  representing cart position, cart velocity, pole angle, and
                  pole angular velocity respectively.
                - info (dict): Empty info dictionary.
        """
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.state.copy(), {}

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
                  ``[cart_pos, cart_vel, pole_angle, pole_vel]``.
                - reward (float): Shaped reward in the range roughly
                  ``(-26, 1]``; a penalty of ``-25`` is applied on
                  termination.
                - terminated (bool): ``True`` when the pole has fallen or the
                  cart has left the track boundary.
                - truncated (bool): ``True`` when the episode step limit is
                  reached.
                - info (dict): Auxiliary diagnostic information forwarded from
                  the parent environment.
        """
        obs, _reward, terminated, truncated, info = super().step(action)

        cart_pos, cart_vel, pole_angle, pole_vel = obs
        norm_angle    = abs(pole_angle) / 0.2095    # gymnasium failure threshold
        norm_cart_pos = abs(cart_pos)  / 2.4         # gymnasium failure threshold
        norm_cart_vel = min(abs(cart_vel) / 3.0, 1.0)  # soft-cap at 3 m/s

        reward = (
            1.0
            - 0.50 * norm_angle
            - 0.45 * norm_cart_pos
            - 0.10 * norm_cart_vel
            - 0.30 * max(cart_vel * cart_pos, 0) / (2.4 * 3.0)
        )
        if terminated:
            reward -= 25.0  

        obs = np.array(self.state, dtype=np.float32)
        return obs, reward, terminated, truncated, info