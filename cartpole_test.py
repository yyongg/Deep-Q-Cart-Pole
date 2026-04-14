import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv
# 1. Create the environment
# 'render_mode' allows us to see the agent later

class MyCustomCartPole(CartPoleEnv):
    def __init__(self, render_mode='human'):
        # Initialize the original CartPole first
        super().__init__(render_mode=render_mode)
        
        # Now, change whatever you want!
        self.gravity = 1.0  # Make it harder
        self.length = .5  # Shorter pole
        self.max_episode_steps = 25  # Longer episodes
    def reset(self, seed=None, options=None):
        # 1. Handle the random seed (important for reproducibility)
        super().reset(seed=seed)
        
        # 2. Define our custom starting state
        # Format: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.state = np.array([0, 0, np.pi, 0]) 
        
        # 3. Return the state as a float32 array and an empty info dictionary
        return np.array(self.state, dtype=np.float32), {}
    def step(self, action):
        # 1. Run the original physics and get the default results
        next_obs, reward, terminated, truncated, info = super().step(action)
        
        # 2. Extract the values we want to check from the observation
        # obs = [cart_pos, cart_vel, pole_angle, pole_vel]
        cart_pos = next_obs[0]
        pole_angle = next_obs[2]
        pole_vel = next_obs[3]
        cart_vel = next_obs[1]
        # 3. Define your own termination logic
        # For example, what if we wanted the game to end if the cart 
        # goes too far to the right (say, 1.0)?
        terminated = False
        if abs(cart_pos) > 2.4: 
            if cart_vel > 1:
                reward -= 25
            terminated = True
        reward += 1 - .9 * abs(pole_angle) - .1* abs(cart_pos)
        print(reward)
        return next_obs, reward, terminated, truncated, info

env = MyCustomCartPole()
# 2. Instantiate the agent
# MlpPolicy means a standard Multi-layer Perceptron (Neural Network)
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the agent
print("Training started...")

try: 
    model.learn(total_timesteps=10000)
except Exception as e:
    print(f"An error occurred: {e}")
print("Training finished!")

# 4. Save the model
model.save("ppo_cartpole")

# 5. Watch the agent play
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()