import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv


# 1. Define the Custom Environment
class MyCustomCartPole(CartPoleEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        # Physics Adjustments
        self.gravity = 2  # Standard gravity is usually better for swing-up momentum
        self.length = 0.25  # Pole length
        self.max_episode_steps = 100  # CRITICAL: Needs time to swing back and forth
        self.force_mag = 15.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start pointing DOWN (pi) with a tiny bit of random noise
        # [cart_pos, cart_vel, pole_angle, pole_vel]
        self.state = np.array([0, 0, np.pi + np.random.uniform(-0.1, 0.1), 0])
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        # 1. Run physics
        next_obs, _, _, truncated, info = super().step(action)

        cart_pos = next_obs[0]
        pole_angle = next_obs[2]
        pole_vel = next_obs[3]

        # 2. Normalize angle to [-pi, pi] where 0 is UP
        normalized_angle = ((pole_angle + np.pi) % (2 * np.pi)) - np.pi

        # 3. QUADRATIC REWARD
        # - Angle is the priority (squared to punish being far away more heavily)
        # - Cart position penalty to keep it from running off-screen
        # - Survival bonus (+1) to prevent the "suicide" strategy
        reward = 3.0 - (normalized_angle**2) - abs(cart_pos)

        # 4. Termination logic
        # Only end if the cart goes off the rails
        terminated = bool(abs(cart_pos) > 2.4)
        if terminated:
            reward -= 5
            terminated = True

        return next_obs, reward, terminated, truncated, info


# Helper function for vectorization
def make_env(rank, seed=0):
    def _init():
        env = MyCustomCartPole(render_mode=None)
        return env

    return _init


if __name__ == "__main__":
    # 2. Setup Vectorized Training
    num_cpu = 4  # Adjust based on your CPU cores
    train_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    train_env = VecMonitor(train_env)

    # 3. Instantiate Agent
    # We use a larger n_steps to gather more data before updating
    model = PPO("MlpPolicy", train_env, verbose=1, n_steps=1024, batch_size=64)

    # 4. Train the agent
    print("Training started... (Targeting 500k steps for a solid swing-up)")
    model.learn(total_timesteps=500_000)
    print("Training finished!")

    # 5. Save the model
    model.save("ppo_cartpole_swingup")

    # 6. Watch the agent play
    print("Watching the results...")
    eval_env = MyCustomCartPole(render_mode="human")
    obs, info = eval_env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        eval_env.render()
        if terminated or truncated:
            obs, info = eval_env.reset()

    eval_env.close()
