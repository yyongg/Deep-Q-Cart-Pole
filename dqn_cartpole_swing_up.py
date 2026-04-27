import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv
from stable_baselines3 import DQN
import math
# 1. Define the Custom Environment (Ensure this matches your original physics!)
class CustomCartPole(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.gravity = 9.81   
        self.length = 0.4    
        self.max_episode_steps = 500 

        high = np.array(
            [
                4.8,                       # cart_pos 
                np.finfo(np.float32).max,  # cart_vel
                1.0,                       # cos(pole_angle) max is 1
                1.0,                       # sin(pole_angle) max is 1
                np.finfo(np.float32).max,  # pole_vel
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
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
        next_obs, _, _, truncated, info = super().step(action)
        cart_pos = next_obs[0]
        cart_vel = next_obs[1]
        pole_angle = next_obs[2]
        pole_vel = next_obs[3]
        normalized_angle = abs(((pole_angle + np.pi) % (2 * np.pi)) - np.pi)
        normalized_pos = abs(cart_pos)*np.pi/2.4
        total = (np.cos(normalized_angle)+1)/2 * max(0, np.cos(normalized_pos)) * np.cos(min(np.pi, (pole_vel)/10))
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
    def _init():
        return CustomCartPole(render_mode=None)
    return _init

if __name__ == "__main__":
    # 2. Setup Vectorized Training
    env = CustomCartPole()
    # 3. LOAD THE WEIGHTS
    # We point PPO to the .zip file and attach it to our current train_env
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
        print("Error: Could not find ppo_cartpole_swingup.zip. Starting from scratch instead.")

    # 4. Fine-tune the agent
    print("Resuming training...")
    model.learn(total_timesteps=300_000) # Adjusted steps for fine-tuning
    print("Training finished!")

    # 5. Save as a NEW verssion
    model.save("dqn_swing_up")
    print("saved")
    # 6. Watch the agent play
    print("Watching the results...")
    eval_env = MyCustomCartPole(render_mode='human')
    obs, info = eval_env.reset()
    
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        eval_env.render()
        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_env.close()