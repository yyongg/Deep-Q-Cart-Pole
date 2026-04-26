"""
carpole_train_yong.py — Train a PPO agent on a custom CartPole env and save the model.
This trains a model to learn how to respond to random nudges from the user rather than
a model that learns how to swing up (cartpole_test.py).
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control import CartPoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class CustomCartPole(CartPoleEnv):
    """CartPole with lower gravity, random velocity kicks during training,
    and a dense shaping reward.  The pending_nudge slot lets the UI inject
    impulses that the environment will consume on the next step."""

    # Match this to NUDGE_STRENGTH in cartpole_interface.py so the model sees realistic kicks.
    DISTURBANCE_MAG = 1.5

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.gravity = 9.81
        self.length = 0.5
        self.pending_nudge = 0.0
        self._sustained_force = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 30% of resets: start with significant cart velocity (the exact failure mode)
        if np.random.rand() < 0.3:
            self.state = np.array(
                [
                    np.random.uniform(-0.8, 0.8),  # cart pos — off center
                    np.random.uniform(-1.5, 1.5),  # cart vel — already moving
                    np.random.uniform(-0.05, 0.05),  # pole near upright
                    np.random.uniform(-0.2, 0.2),
                ],
                dtype=np.float32,
            )
        else:
            self.state = np.array(
                [
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.3, 0.3),
                    np.random.uniform(-0.03, 0.03),
                    np.random.uniform(-0.1, 0.1),
                ],
                dtype=np.float32,
            )

        self._sustained_force = 0.0
        self.pending_nudge = 0.0
        return self.state.copy(), {}

    def step(self, action):
        self.state[1] += self.pending_nudge
        self.pending_nudge = 0.0

        if np.random.rand() < 0.08:
            self.state[1] += np.random.uniform(
                -self.DISTURBANCE_MAG, self.DISTURBANCE_MAG
            )
        if np.random.rand() < 0.04:
            self.state[3] += np.random.uniform(-0.5, 0.5)

        obs, _reward, terminated, truncated, info = super().step(action)
        cart_pos, cart_vel, pole_angle, pole_vel = obs

        # Velocity termination — makes the accelerating strategy fatal
        if abs(cart_vel) > 2.5:
            terminated = True

        p = cart_pos / 2.4
        v = cart_vel / 2.5
        a = pole_angle / 0.2095

        diverging = float(np.sign(cart_vel) == np.sign(cart_pos))

        reward = 1.0 - 1.5 * a**2 - 2.0 * p**2 - 1.5 * v**2 - 1.0 * v**2 * diverging

        if terminated:
            reward -= 5.0

        obs = np.array(self.state, dtype=np.float32)
        return obs, reward, terminated, truncated, info


def make_env(render_mode=None):
    env = CustomCartPole(render_mode=render_mode)
    return TimeLimit(env, max_episode_steps=500)  # longer episodes → harder


if __name__ == "__main__":
    # Vectorised envs speed up sample collection significantly.
    N_ENVS = 4
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=2e-4,  # slightly lower for stability
        n_steps=2048,
        batch_size=128,  # larger batch helps with noisy rewards
        n_epochs=10,
        gamma=0.995,  # higher — agent should care about being centered long-term
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # less entropy needed once env is harder
        policy_kwargs=dict(net_arch=[256, 256]),  # bigger net for more complex task
    )

    print("Training started …")
    model.learn(total_timesteps=1_000_000)

    print("Training finished!")

    model.save("ppo_cartpole_yong_4-26")
    print("Model saved to ppo_cartpole.zip")
    vec_env.close()

    eval_env = make_env(render_mode="human")
    obs, _ = eval_env.reset()

    for _ in range(2_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        if terminated or truncated:
            obs, _ = eval_env.reset()

    eval_env.close()
