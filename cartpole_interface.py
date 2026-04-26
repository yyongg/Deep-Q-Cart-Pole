"""
cartpole_interface.py — Interactive CartPole viewer.

Load the trained PPO model and let the user nudge the pole with arrow keys.
This file contains NO training logic — it is inference-only.

Controls:
  LEFT  arrow  → kick the pole leftward  (negative angular velocity impulse)
  RIGHT arrow  → kick the pole rightward (positive angular velocity impulse)
  R            → reset the episode manually
  Q / Escape   → quit
"""

import numpy as np
import pygame
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control import CartPoleEnv
from stable_baselines3 import PPO


class CustomCartPole(CartPoleEnv):
    """Matches the physics used during training (gravity, pole length).
    No random disturbances here — those were only needed for training."""

    def __init__(self):
        super().__init__(render_mode="rgb_array")
        self.gravity = 9.8
        self.length = 0.5
        self.masspole = 0  # default values

    def set_parameters(self, length, weight):
        """
            Overides the length and weight with the given user input.

        Args:
            length: An integer that represents the length of the pole in meters

            weight: An integer that represents the mass of the pole in kg

            Returns:
                None - sets class attributes of self.length and self.masspole to length and weight respectively
        """
        self.length = length
        self.masspole = weight

    def reset(self, seed=None, options=None):
        """Reset the environment to a fixed upright starting state.

        Overrides the default random initialisation so that every episode
        begins with the cart and pole perfectly stationary at the origin.

        Args:
            seed (int | None): Random seed forwarded to the parent reset.
                Has no practical effect here because the state is hard-coded,
                but is accepted for API compatibility.
            options (dict | None): Currently unused; accepted for API
                compatibility with the Gymnasium interface.

        Returns:
            tuple[np.ndarray, dict]: A tuple of:
                - obs (np.ndarray): Initial observation ``[0, 0, 0, 0]``
                  representing cart position, cart velocity, pole angle, and
                  pole angular velocity respectively.
                - info (dict): Empty info dictionary.
        """
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        """Advance the simulation by one timestep with a shaped reward.

        Applies the given action via the parent ``step``, then replaces the
        default binary (+1 / episode-end) reward with a dense signal that
        penalises pole tilt, cart displacement, cart speed, and momentum
        directed away from the centre.

        Args:
            action (int): Discrete action — ``0`` to push the cart left,
                ``1`` to push the cart right.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: A tuple of:
                - obs (np.ndarray): Current observation
                  ``[cart_pos, cart_vel, pole_angle, pole_vel]``.
                - reward (float): Shaped reward in the range roughly
                  ``(-26, 1]``; a penalty of ``-25`` is applied on
                  termination.
                - terminated (bool): ``True`` when the pole has fallen or the
                  cart has left the track boundary.
                - truncated (bool): ``True`` when the episode step limit is
                  reached.
                - info (dict): Auxiliary diagnostic information forwarded from
                  the parent environment.
        """
        obs, _reward, terminated, truncated, info = super().step(action)

        cart_pos, cart_vel, pole_angle, pole_vel = obs
        norm_angle = abs(pole_angle) / 0.2095  # gymnasium failure threshold
        norm_cart_pos = abs(cart_pos) / 2.4  # gymnasium failure threshold
        norm_cart_vel = min(abs(cart_vel) / 3.0, 1.0)  # soft-cap at 3 m/s

        reward = (
            1.0
            - 0.50 * norm_angle
            - 0.45 * norm_cart_pos
            - 0.10 * norm_cart_vel
            - 0.30 * max(cart_vel * cart_pos, 0) / (2.4 * 3.0)
        )
        if terminated:
            reward -= 25.0

        obs = np.array(self.state, dtype=np.float32)
        return obs, reward, terminated, truncated, info


SCREEN_W, SCREEN_H = 800, 800
FPS = 60
NUDGE_STRENGTH = 0.5
NUDGE_DISPLAY_TTL = 20

# Colour palette
BG_TOP = (15, 20, 35)
BG_BOT = (30, 40, 65)
WHITE = (240, 245, 255)
YELLOW = (255, 215, 50)
RED = (220, 60, 60)
GREEN = (60, 200, 100)
GREY = (120, 130, 150)
PANEL_BG = (0, 0, 0, 160)  # RGBA for transparent panel

# Thresholds (matching gymnasium defaults)
MAX_ANGLE = 0.2095
MAX_CART = 2.4


def lerp_color(a, b, t):
    """Linearly interpolate between two RGB colours.

    Args:
        a (tuple[int, int, int]): Starting RGB colour, e.g. ``(255, 0, 0)``.
        b (tuple[int, int, int]): Ending RGB colour, e.g. ``(0, 255, 0)``.
        t (float): Interpolation factor in ``[0.0, 1.0]`` where ``0.0``
            returns ``a`` and ``1.0`` returns ``b``.

    Returns:
        tuple[int, int, int]: Interpolated RGB colour with integer components.
    """
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def bar_color(fraction):
    """Map a normalised fraction to a green → yellow → red colour gradient.

    Args:
        fraction (float): Value in ``[0.0, 1.0]`` representing how close a
            sensor reading is to its danger threshold. ``0.0`` is safe
            (green) and ``1.0`` is critical (red).

    Returns:
        tuple[int, int, int]: RGB colour corresponding to the given fraction.
    """
    if fraction < 0.5:
        return lerp_color(GREEN, YELLOW, fraction * 2)
    return lerp_color(YELLOW, RED, (fraction - 0.5) * 2)


def draw_gradient_bg(surface):
    """Fill a surface with a vertical top-to-bottom gradient background.

    Draws a gradient from ``BG_TOP`` at the top edge to ``BG_BOT`` at the
    bottom edge, one horizontal line at a time.

    Args:
        surface (pygame.Surface): The Pygame surface to draw onto. Modified
            in place.

    Returns:
        None
    """
    for y in range(SCREEN_H):
        t = y / SCREEN_H
        color = lerp_color(BG_TOP, BG_BOT, t)
        pygame.draw.line(surface, color, (0, y), (SCREEN_W, y))


def draw_status_bar(surface, font_sm, label, value, max_val, x, y, w=160, h=14):
    """Draw a labelled horizontal progress bar at the specified position.

    The filled portion of the bar is coloured via ``bar_color`` based on how
    close ``value`` is to ``max_val``. A text label showing the raw value is
    rendered immediately above the bar.

    Args:
        surface (pygame.Surface): The Pygame surface to draw onto.
        font_sm (pygame.font.Font): Small font used to render the label text.
        label (str): Short description shown before the numeric value,
            e.g. ``"Cart "`` or ``"Angle"``.
        value (float): Current sensor reading. Its absolute value is compared
            against ``max_val`` to compute the fill fraction.
        max_val (float): The threshold at which the bar becomes fully filled
            (fraction = 1.0). Must be non-zero.
        x (int): Left edge x-coordinate of the bar in pixels.
        y (int): Top edge y-coordinate of the bar in pixels.
        w (int): Width of the bar in pixels. Defaults to ``160``.
        h (int): Height of the bar in pixels. Defaults to ``14``.

    Returns:
        None
    """
    fraction = min(abs(value) / max_val, 1.0)
    color = bar_color(fraction)

    pygame.draw.rect(surface, (40, 50, 70), (x, y, w, h), border_radius=4)
    if fraction > 0:
        pygame.draw.rect(surface, color, (x, y, int(w * fraction), h), border_radius=4)
    pygame.draw.rect(surface, GREY, (x, y, w, h), 1, border_radius=4)

    lbl = font_sm.render(f"{label}: {value:+.3f}", True, WHITE)
    surface.blit(lbl, (x, y - 18))


def draw_hud(
    surface,
    fonts,
    episode,
    step,
    total_reward,
    cart_pos,
    pole_angle,
    nudge_ttl,
    last_nudge_dir,
):
    """Draw the full HUD overlay onto the given surface.

    Renders a semi-transparent stats panel (episode, step, cumulative
    reward), two sensor progress bars (cart position, pole angle), a
    fading nudge indicator when the user has recently applied a kick, and
    a keyboard-controls legend in the bottom-right corner.

    Args:
        surface (pygame.Surface): The Pygame surface to draw onto.
        fonts (tuple[pygame.font.Font, pygame.font.Font]): A 2-tuple of
            ``(font_lg, font_sm)`` where ``font_lg`` is used for stats and
            nudge text, and ``font_sm`` is used for bar labels and the
            controls legend.
        episode (int): Current episode number, displayed in the stats panel.
        step (int): Current step count within the episode.
        total_reward (float): Cumulative shaped reward accumulated so far in
            the current episode.
        cart_pos (float): Current cart position in metres, used to fill the
            cart-position sensor bar.
        pole_angle (float): Current pole angle in radians, used to fill the
            angle sensor bar.
        nudge_ttl (int): Remaining display frames for the nudge indicator.
            The indicator fades out as this value approaches ``0``.
        last_nudge_dir (int): Direction of the most recent nudge: ``-1`` for
            left, ``1`` for right. Used to choose the indicator label.

    Returns:
        None
    """
    font_lg, font_sm = fonts
    pad = 14

    # --- semi-transparent panel ---
    panel = pygame.Surface((200, 160), pygame.SRCALPHA)
    panel.fill(PANEL_BG)
    surface.blit(panel, (pad, pad))

    stats = [
        f"Episode : {episode}",
        f"Step    : {step}",
        f"Reward  : {total_reward:+.2f}",
    ]
    for i, line in enumerate(stats):
        shadow = font_lg.render(line, True, (0, 0, 0))
        text = font_lg.render(line, True, WHITE)
        surface.blit(shadow, (pad + 9, pad + 9 + i * 26))
        surface.blit(text, (pad + 8, pad + 8 + i * 26))

    # --- sensor bars ---
    bar_x = pad + 8
    bar_y = pad + 100
    draw_status_bar(surface, font_sm, "Cart ", cart_pos, MAX_CART, bar_x, bar_y)
    draw_status_bar(surface, font_sm, "Angle", pole_angle, MAX_ANGLE, bar_x, bar_y + 42)

    # --- nudge indicator ---
    if nudge_ttl > 0:
        alpha = int(255 * (nudge_ttl / NUDGE_DISPLAY_TTL))
        label = "◀  NUDGE LEFT" if last_nudge_dir < 0 else "NUDGE RIGHT  ▶"
        nudge_surf = font_lg.render(label, True, YELLOW)
        nudge_surf.set_alpha(alpha)
        nx = SCREEN_W // 2 - nudge_surf.get_width() // 2
        surface.blit(nudge_surf, (nx, SCREEN_H - 44))

    # --- controls legend (bottom-right) ---
    legend = ["← → : nudge pole", "R : reset", "Q / Esc : quit"]
    for i, line in enumerate(legend):
        t = font_sm.render(line, True, GREY)
        surface.blit(
            t, (SCREEN_W - t.get_width() - pad, SCREEN_H - pad - (len(legend) - i) * 20)
        )


def main(length, weight):
    """Run the interactive CartPole visualisation loop.

    Loads the trained PPO model from ``ppo_cartpole_yong_4-26.zip``, opens a Pygame
    window, and enters the main game loop. On each frame the function:

    1. Polls keyboard events — handling quit, manual reset, and held
       left/right keys for pole nudging.
    2. Applies any angular-velocity nudge directly to the environment state.
    3. Queries the PPO policy for an action and steps the environment.
    4. Renders the cart-pole frame, scales it to fill the window, and
       overlays the HUD.
    5. Automatically resets when an episode terminates or is truncated.

    Returns:
        None

    Raises:
        FileNotFoundError: If ``ppo_cartpole.zip`` cannot be found in the
            current working directory when ``PPO.load`` is called.
        pygame.error: If Pygame fails to initialise the display or font
            subsystems (e.g. no available video driver).
    """
    base_env = CustomCartPole()
    env = TimeLimit(base_env, max_episode_steps=500)
    base_env.set_parameters(length, weight)
    model = PPO.load("ppo_cartpole_yong_4-26", env=env)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("CartPole — Arrow keys to nudge")
    font_lg = pygame.font.SysFont("monospace", 17, bold=True)
    font_sm = pygame.font.SysFont("monospace", 13)
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    episode = 1
    step_count = 0
    total_reward = 0.0
    nudge_ttl = 0
    last_nudge_dir = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                    step_count = 0
                    total_reward = 0.0

        pole_nudge = 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pole_nudge = -NUDGE_STRENGTH
            nudge_ttl = NUDGE_DISPLAY_TTL
            last_nudge_dir = -1
        elif keys[pygame.K_RIGHT]:
            pole_nudge = NUDGE_STRENGTH
            nudge_ttl = NUDGE_DISPLAY_TTL
            last_nudge_dir = 1

        env.unwrapped.state[3] += pole_nudge
        obs = np.array(env.unwrapped.state, dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        step_count += 1
        if nudge_ttl > 0:
            nudge_ttl -= 1

        draw_gradient_bg(screen)

        frame = env.render()
        frame = np.transpose(frame, (1, 0, 2))
        cart_surface = pygame.surfarray.make_surface(frame)

        cart_surface = pygame.transform.scale(cart_surface, (SCREEN_W, SCREEN_H))
        screen.blit(cart_surface, (0, 0))

        cart_pos, _, pole_angle, _ = obs
        draw_hud(
            screen,
            (font_lg, font_sm),
            episode,
            step_count,
            total_reward,
            cart_pos,
            pole_angle,
            nudge_ttl,
            last_nudge_dir,
        )

        pygame.display.flip()

        if terminated or truncated:
            obs, _ = env.reset()
            episode += 1
            step_count = 0
            total_reward = 0.0

        clock.tick(FPS)

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
