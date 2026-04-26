import pygame
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from model import CustomCartPole  # Import your CustomCartPole class

class CartPoleController:
    def __init__(self, view_instance):
        self.view = view_instance
        self.model = None
        self.ai = None
        self.env = None

    def setup_session(self):
        """Phase 1: Get inputs and build the environment."""
        # Use the View's popup to get parameters
        l_text = "Enter Pole Length (m):"
        while True:
            res = self.view.get_input_popup(l_text)
            try:
                length = float(res)
                if length <= 0: raise ValueError
                break
            except ValueError: l_text = "Invalid. Enter Length:"

        w_text = "Enter Weight (kg):"
        while True:
            res = self.view.get_input_popup(w_text)
            try:
                weight = float(res)
                if weight <= 0: raise ValueError
                break
            except ValueError: w_text = "Invalid. Enter Weight:"

        # Initialize the Model with these parameters
        base_env = CustomCartPole()
        base_env.set_parameters(length, weight)
        
        # Wrap for Gymnasium compatibility
        self.env = TimeLimit(base_env, max_episode_steps=500)
        
        # Load the AI model
        try:
            self.ai = PPO.load("ppo_cartpole_yong_4-26", env=self.env)
        except FileNotFoundError:
            print("AI Weight file not found! Simulation will run with random actions.")
            self.ai = None

    def run_simulation(self):
        """Phase 2: The actual simulation loop."""
        obs, _ = self.env.reset()
        episode, step_count, total_reward = 1, 0, 0.0
        nudge_ttl, last_nudge_dir = 0, 0
        
        sim_running = True
        while sim_running:
            # 1. Handle Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "QUIT"
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return "QUIT"
                    if event.key == pygame.K_u:
                        return "RESTART"  # Break loop to go back to inputs
                    if event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        step_count, total_reward = 0, 0.0

            # 2. Process Nudges (Manual Input)
            pole_nudge = 0.0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                pole_nudge = -self.view.NUDGE_STRENGTH
                nudge_ttl, last_nudge_dir = self.view.NUDGE_DISPLAY_TTL, -1
            elif keys[pygame.K_RIGHT]:
                pole_nudge = self.view.NUDGE_STRENGTH
                nudge_ttl, last_nudge_dir = self.view.NUDGE_DISPLAY_TTL, 1

            # Apply nudge to the Model
            self.env.unwrapped.state[3] += pole_nudge
            obs = np.array(self.env.unwrapped.state, dtype=np.float32)

            # 3. AI Prediction
            if self.ai:
                action, _ = self.ai.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()

            # 4. Step the Model
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            step_count += 1
            if nudge_ttl > 0: nudge_ttl -= 1

            # 5. Tell the View to Render
            self.view.draw_gradient_bg(self.view._screen)
            
            # Get the raw frame from the Model
            frame = self.env.render()
            frame = np.transpose(frame, (1, 0, 2))
            cart_surface = pygame.surfarray.make_surface(frame)
            cart_surface = pygame.transform.scale(cart_surface, (800, 800))
            self.view._screen.blit(cart_surface, (0, 0))

            # Draw the HUD
            cart_pos, _, pole_angle, _ = obs
            self.view.draw_hud(
                self.view._screen, 
                (self.view._font, self.view._font), # Assuming standard font
                episode, step_count, total_reward,
                cart_pos, pole_angle, nudge_ttl, last_nudge_dir
            )

            pygame.display.flip()

            if terminated or truncated:
                obs, _ = self.env.reset()
                episode += 1
                step_count, total_reward = 0, 0.0

            self.view._clock.tick(self.view.FPS)