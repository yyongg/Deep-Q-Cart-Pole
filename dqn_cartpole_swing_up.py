"""
This module defines a custom reinforcement learning environment and training pipeline 
for the Cart-Pole swing-up task. It utilizes Gymnasium for physics simulation and 
Stable Baselines3 for the Deep Q-Network (DQN) implementation.
"""

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv
from stable_baselines3 import DQN
import math

class CustomCartPole(CartPoleEnv):
    """
    An extension of the Gymnasium CartPole environment modified to support 
    a swing-up task. It uses a continuous observation space including 
    trigonometric representations of the pole angle to avoid discontinuities.
    """
    def __init__(self, render_mode=None):
        """
        Initializes the environment with custom physical constants and a 
        5-dimensional observation space: [cart_pos, cart_vel, cos(theta), 
        sin(theta), pole_vel].
        """
        super().__init__(render_mode=render_mode)
        self.gravity = 9.81   
        self.length = 0.4    
        self.max_episode_steps = 500 

        high = np.array(
            [
                4.8,                       
                np.finfo(np.float32).max,  
                1.0,                       
                1.0,                       
                np.finfo(np.float32).max,  
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment state such that the pole begins hanging 
        downward (pi radians) with a small amount of random noise.
        """
        super().reset(seed=seed)
        self.state = np.array([0, 0, np.pi + np.random.uniform(-0.1, 0.1), 0]) 

        obs = np.array([
            self.state[0], 
            self.state[1], 
            np.cos(self.state[2]), 
            np.sin(self.state[2]), 
            self.state[3]
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        """
        Processes a single simulation step. Replaces the default binary 
        reward with a shaped, dense reward function designed to encourage 
        the agent to swing the pole up and balance it at the center.
        """
        next_obs, _, _, truncated, info = super().step(action)
        cart_pos = next_obs[0]
        cart_vel = next_obs[1]
        pole_angle = next_obs[2]
        pole_vel = next_obs[3]
        
        normalized_angle = abs(((pole_angle + np.pi) % (2 * np.pi)) - np.pi)
        normalized_pos = abs(cart_pos) * np.pi / 2.4
        
        total = (np.cos(normalized_angle) + 1) / 2 * max(0, np.cos(normalized_pos)) * np.cos(min(np.pi, (pole_vel) / 10))
        reward = total - .1
        
        terminated = bool(abs(cart_pos) > 2.4)
        if terminated: 
            reward -= 5
            terminated = True
            
        obs = np.array([
            cart_pos,
            cart_vel,
            np.cos(pole_angle),
            np.sin(pole_angle),
            pole_vel
        ])
        return obs, reward, terminated, truncated, info

def make_env(rank, seed=0):
    """
    A helper utility that generates an instance of the CustomCartPole 
    environment, typically used for vectorized training processes.
    """
    def _init():
        return CustomCartPole(render_mode=None)
    return _init

if __name__ == "__main__":
    """
    Executes the main training and evaluation sequence. This includes 
    instantiating the environment, configuring the DQN agent with 
    specific hyperparameters, and running the learning loop.
    """
    env = CustomCartPole()
    
    try:
        model = DQN(
            "MlpPolicy", 
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=200000,   
            learning_starts=1000,   
            tensorboard_log="./dqn_cartpole_tensorboard/", 
            batch_size=64,      
            tau=0.5,               
            target_update_interval=500, 
            exploration_fraction=0.1,
            exploration_final_eps=0.05 
            )
    except FileNotFoundError:
        print("Error: Could not find model configuration. Starting from scratch instead.")

    print("Resuming training...")
    model.learn(total_timesteps=300_000) 
    
    model.save("dqn_swing_up")
    print("Model saved successfully.")

    print("Watching the results...")
    eval_env = CustomCartPole(render_mode='human')
    obs, info = eval_env.reset()
    
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        eval_env.render()
        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_env.close()