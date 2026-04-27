"""
MVC - Controller

This module is the controller for the Cart-Pole simulation. It manages
the simulation, handles the user inputs such as nudges in the simulation
interfaces with the model and updates the class View for rendering.
"""

import os
import pygame
import numpy as np
from stable_baselines3 import DQN
from gymnasium.wrappers import TimeLimit
from model import CustomCartPole


class CartPoleController:
    """
    Manages the interaction between the Cart-Pole Model and View.

    The controller handles the setup of the reinforcement learning environment,
    loading of neural network weights, and the main simulation execution loop.

    Attributes:
        view (View): An instance of the View class for rendering and UI.
        model (DQN): The trained Stable Baselines3 model for pole balancing.
        env (gym.Env): The wrapped Gymnasium environment (CustomCartPole).
    """

    def __init__(self, view_instance):
        """
        Initializes the controller with a reference to the UI view.
        """
        self.view = view_instance
        self.model = None
        self.env = None

    def get_user_inputs(self):
        """
        Uses the View to prompt the user for the pole's physical parameters.

        Returns:
            tuple[float, float]: The chosen length (in feet) and weight (in kg).
        """
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

        return length, weight

    def setup_env_and_model(self, length, weight):
        """
        Initializes the environment with the user's parameters and loads the model.

        Args:
            length (float): Pole length in feet.
            weight (float): Pole mass in kg.

        Returns:
            bool: True if a new model was created and needs training, False if loaded.
        """
        base_env = CustomCartPole(render_mode="rgb_array")
        base_env.set_parameters(length * 0.3048, weight)
        self.env = TimeLimit(base_env, max_episode_steps=500)

        model_path = f"dqn_swing_up_{length}_{weight}.zip"

        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}...")
            self.model = DQN.load(model_path, env=self.env)
            return False  # No training required
        else:
            print("No model found. Initializing a new DQN model...")
            self.model = DQN(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=1e-3,
                buffer_size=150000,
                learning_starts=1000,
                batch_size=128,
                tau=0.5,
                target_update_interval=500,
                exploration_fraction=0.1,
                exploration_final_eps=0.05,
            )
            return True  # Training required!

    def nudges(self, nudge_ttl, last_nudge_dir):
        """
        Checks for user arrow-key input to apply an instant velocity kick.
        """
        pole_nudge = 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pole_nudge = -self.view.thresholds["nudge_strength"]
            nudge_ttl, last_nudge_dir = self.view.thresholds["nudge_display_ttl"], -1
        elif keys[pygame.K_RIGHT]:
            pole_nudge = self.view.thresholds["nudge_strength"]
            nudge_ttl, last_nudge_dir = self.view.thresholds["nudge_display_ttl"], 1

        self.env.unwrapped.state[3] += pole_nudge
        obs = np.array(self.env.unwrapped.state, dtype=np.float32)

        formatted_obs = np.array(
            [obs[0], obs[1], np.cos(obs[2]), np.sin(obs[2]), obs[3]], dtype=np.float32
        )

        if self.model:
            action, _ = self.model.predict(formatted_obs, deterministic=True)
        else:
            action = self.env.action_space.sample()

        return action, nudge_ttl, last_nudge_dir

    def run_simulation(self):
        """
        Executes the main interactive simulation loop.
        """
        obs, _ = self.env.reset()
        nudge_list = [1, 0, 0.0, 0, 0]

        sim_running = True
        while sim_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "QUIT"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "QUIT"
                    if event.key == pygame.K_r:
                        return "RESTART"
                    if event.key == pygame.K_t:
                        obs, _ = self.env.reset()
                        nudge_list[1], nudge_list[2] = 0, 0.0

            action, nudge_list[3], nudge_list[4] = self.nudges(
                nudge_list[3], nudge_list[4]
            )
            obs, reward, terminated, truncated, _ = self.env.step(action)
            nudge_list[2] += reward
            nudge_list[1] += 1
            if nudge_list[3] > 0:
                nudge_list[3] -= 1

            self.view.draw_gradient_bg(self.view.screen)

            frame = self.env.render()
            frame = np.transpose(frame, (1, 0, 2))
            cart_surface = pygame.surfarray.make_surface(frame)
            cart_surface = pygame.transform.scale(cart_surface, (800, 800))
            self.view.screen.blit(cart_surface, (0, 0))

            cart_pos = obs[0]
            cos_angle = obs[2]
            sin_angle = obs[3]
            pole_angle = np.arctan2(sin_angle, cos_angle)

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

            if terminated or truncated:
                obs, _ = self.env.reset()
                nudge_list[0] += 1
                nudge_list[1], nudge_list[2] = 0, 0.0

            self.view.clock.tick(self.view.fps)
        return None
