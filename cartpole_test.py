import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the environment
# 'render_mode' allows us to see the agent later
env = gym.make("CartPole-v1", render_mode="human")

# 2. Instantiate the agent
# MlpPolicy means a standard Multi-layer Perceptron (Neural Network)
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the agent
print("Training started...")
model.learn(total_timesteps=10000)
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