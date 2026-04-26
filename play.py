import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
# Assuming your MyCustomCartPole is defined in this file or imported
from gymnasium.envs.classic_control import CartPoleEnv

class MyCustomCartPole(CartPoleEnv):
    def __init__(self, render_mode='human'):
        super().__init__(render_mode=render_mode)
        self.gravity = 9.8
        self.length = 0.75
        self.max_episode_steps = 500
        self.masspole = 10
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Match your training starting state!
        self.state = np.array([0, 0, np.pi, 0]) 
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        next_obs, reward, terminated, truncated, info = super().step(action)
        # Note: You don't actually need the reward logic here just to watch it play,
        # but keeping the termination logic consistent is good practice.
        cart_pos = next_obs[0]
        terminated = bool(abs(cart_pos) > 2.4)
        return next_obs, reward, terminated, truncated, info

# 1. Initialize the environment with 'human' render mode
env = MyCustomCartPole(render_mode='human')

# 2. Load the trained agent
# Ensure the path matches the filename you saved earlier
model = PPO.load("ppo_cartpole_swingup_v2")

print("Model loaded! Press Ctrl+C to stop.")

# 3. Running loop
obs, info = env.reset()
try:
    while True:
        # Use deterministic=True for the 'best' performance
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment handles rendering automatically because of 'human' mode
        if terminated or truncated:
            obs, info = env.reset()
            print("Episode finished, resetting...")

except KeyboardInterrupt:
    print("\nClosing environment.")
finally:
    env.close()