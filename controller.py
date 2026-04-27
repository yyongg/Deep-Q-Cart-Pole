"""
MVC - Controller

This module is the controller for the Cart-Pole simulation. It manages
the simulation, handles the user inputs such as nudges in the simulation
interfaces with the model and updates the class View for rendering.
"""

import pygame
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from model import CustomCartPole  # Import your CustomCartPole class


class CartPoleController:
    """
    Manages the interaction between the Cart-Pole Model and View.

    The controller handles the setup of the reinforcement learning environment,
    loading of neural network weights, and the main simulation execution loop.

    Attributes:
        view (View): An instance of the View class for rendering and UI.
        model (PPO): The trained Stable Baselines3 model for pole balancing.
        env (gym.Env): The wrapped Gymnasium environment (CustomCartPole).
    """

    def __init__(self, view_instance):
        """
        Initializes the controller with a reference to the UI view.

        Args:
            view_instance (View): The View object that manages the display.
        Returns:
            None
        """
        self.view = view_instance
        self.model = None
        self.env = None

    def setup_session(self):
        """
        Asks the user for the simulation parameters and initializes the model.
        This collects pole length and weight by calling the view class function get_input_popup
        and attempts to load the pre-trained PPO model weights.

        Args:
            None
        Return:
            None
        Raises:
            FileNotFoundError: If the specified model weights file is missing.
        """
        # Gets user inputs
        l_text = "Enter Pole Length (ft):"
        while True:
            res = self.view.get_input_popup(l_text)
            try:
                length = float(res)
                if length <= 0:
                    raise ValueError
                break
            except ValueError:
                l_text = "Invalid input"

        w_text = "Enter Weight (kg):"
        while True:
            res = self.view.get_input_popup(w_text)
            try:
                weight = float(res)
                if weight <= 0:
                    raise ValueError
                break
            except ValueError:
                w_text = "Invalid input"

        # Initialize the model
        base_env = CustomCartPole()
        base_env.set_parameters(
            length * 0.3048, weight
        )  # convert inputted(ft) to meters
        self.env = TimeLimit(base_env, max_episode_steps=500)

        # Load the weight files from trained model
        try:
            self.model = PPO.load("ppo_cartpole_yong_4-26", env=self.env)
        except FileNotFoundError:
            print("File not found")
            self.model = None

    def nudges(self, nudge_ttl, last_nudge_dir):
        """
        This function checks for user arrow-key input to apply an instant velocity
        'kick' to the pole. It updates the environment's internal state based on this

        Args:
            nudge_ttl (int): The current 'Time To Live' (frames remaining) for
                the nudge visual indicator.
            last_nudge_dir (int): The direction of the last applied nudge
                (-1 for left, 1 for right, 0 for none).

        Returns:
            tuple[int, int, int]: A tuple containing:
                - action (int): The discrete action (0 or 1) determined by
                  the model or random sampling.
                - nudge_ttl (int): Updated display frames for the nudge UI.
                - last_nudge_dir (int): The direction of the nudge applied
                  during this frame.
        """
        pole_nudge = 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  # pylint: disable=no-member
            pole_nudge = -self.view.thresholds["nudge_strength"]
            nudge_ttl, last_nudge_dir = self.view.thresholds["nudge_display_ttl"], -1
        elif keys[pygame.K_RIGHT]:  # pylint: disable=no-member
            pole_nudge = self.view.thresholds["nudge_strength"]
            nudge_ttl, last_nudge_dir = self.view.thresholds["nudge_display_ttl"], 1

        self.env.unwrapped.state[3] += pole_nudge
        obs = np.array(self.env.unwrapped.state, dtype=np.float32)
        if self.model:
            action, _ = self.model.predict(obs, deterministic=True)
        else:
            action = self.env.action_space.sample()
        return action, nudge_ttl, last_nudge_dir

    def run_simulation(self):
        """
        Executes the main simulation loop which handles user nudges (Arrow keys), model predictions,
        physics updates, and view rendering.

        Args:
            None
        Returns:
            A pygame window of the simulation running
        """
        obs, _ = self.env.reset()
        nudge_list = [
            1,
            0,
            0.0,
            0,
            0,
        ]  # index 1: episode , index 2: step_count,
        # index 3: total_reward, index 4: nudge_ttl, index 5: last_nudge_dir

        sim_running = True
        while sim_running:
            # Keyboard inputs for restarting,quiting,etc
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # pylint: disable=no-member
                    return "QUIT"
                if event.type == pygame.KEYDOWN:  # pylint: disable=no-member
                    if event.key == (pygame.K_ESCAPE):  # pylint: disable=no-member
                        return "QUIT"
                    if event.key == pygame.K_r:  # pylint: disable=no-member
                        return "RESTART"  # asks for user inputs again
                    if event.key == pygame.K_r:  # pylint: disable=no-member
                        obs, _ = self.env.reset()
                        nudge_list[1], nudge_list[2] = 0, 0.0

            # nudges
            action, nudge_list[3], nudge_list[4] = self.nudges(
                nudge_list[3], nudge_list[4]
            )
            # physics update
            obs, reward, terminated, truncated, _ = self.env.step(action)
            nudge_list[2] += reward
            nudge_list[1] += 1
            if nudge_list[3] > 0:
                nudge_list[3] -= 1

            # draws background
            self.view.draw_gradient_bg(self.view.screen)

            # envrionment rendering
            frame = self.env.render()
            frame = np.transpose(
                frame, (1, 0, 2)
            )  # sets axes for pygame coordinate system
            cart_surface = pygame.surfarray.make_surface(frame)
            cart_surface = pygame.transform.scale(cart_surface, (800, 800))
            self.view.screen.blit(cart_surface, (0, 0))

            # UI overlay - extracts metrics and draws them on the HUD
            cart_pos, _, pole_angle, _ = obs

            # Package all HUD data into a single dictionary
            hud_data = {
                "episode": nudge_list[0],
                "step": nudge_list[1],
                "reward": nudge_list[2],
                "cart_pos": cart_pos,
                "pole_angle": pole_angle,
                "nudge_ttl": nudge_list[3],
                "last_nudge_dir": nudge_list[4],
            }

            self.view.draw_hud(
                self.view.screen, (self.view.font, self.view.font), hud_data
            )

            pygame.display.flip()

            # resets the program
            if terminated or truncated:
                obs, _ = self.env.reset()
                nudge_list[0] += 1
                nudge_list[1], nudge_list[2] = 0, 0.0

            self.view.clock.tick(self.view.fps)
        return None
