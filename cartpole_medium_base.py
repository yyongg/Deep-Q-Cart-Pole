import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv

# 1. Define the Custom Environment (Ensure this matches your original physics!)
class MyCustomCartPole(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.gravity = 4.0   
        self.length = 0.75    
        self.max_episode_steps = 100 
        self.force_mag = 15.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0, np.pi + np.random.uniform(-0.1, 0.1), 0]) 
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        next_obs, _, _, truncated, info = super().step(action)
        cart_pos = next_obs[0]
        pole_angle = next_obs[2]
        pole_vel = next_obs[3]
        normalized_angle = ((pole_angle + np.pi) % (2 * np.pi)) - np.pi
        reward = 9.0 - (normalized_angle**2) - (cart_pos**2)
        if np.degrees(normalized_angle) < 30 and cart_pos < 1.25: 
            reward+=2-(pole_vel**2)
            
        terminated = bool(abs(cart_pos) > 2.4)
        if terminated: 
            reward -= 5
            terminated = True
        return next_obs, reward, terminated, truncated, info

def make_env(rank, seed=0):
    def _init():
        return MyCustomCartPole(render_mode=None)
    return _init

if __name__ == "__main__":
    # 2. Setup Vectorized Training
    num_cpu = 4  
    train_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    train_env = VecMonitor(train_env)

    # 3. LOAD THE WEIGHTS
    # We point PPO to the .zip file and attach it to our current train_env
    print("Loading existing weights from ppo_cartpole_swingup.zip...")
    try:
        model = PPO.load("ppo_cartpole_swingup", env=train_env)
        # Optional: You can override hyperparameters here if fine-tuning
        # model.learning_rate = 0.0001 
    except FileNotFoundError:
        print("Error: Could not find ppo_cartpole_swingup.zip. Starting from scratch instead.")
        model = PPO("MlpPolicy", train_env, verbose=1, n_steps=1024, batch_size=64)

    # 4. Fine-tune the agent
    print("Resuming training...")
    model.learn(total_timesteps=200_000) # Adjusted steps for fine-tuning
    print("Training finished!")

    # 5. Save as a NEW version
    model.save("ppo_cartpole_swingup_v2")
    print("Saved as ppo_cartpole_swingup_v2")

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