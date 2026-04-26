import pygame
import sys
import numpy as np
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control import CartPoleEnv
from stable_baselines3 import PPO

class view:
    def __init__(self):
        # Sets up pygame stuff
        pygame.init()
        self.screen_width = 800
        self.screen_height = 800
        self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("CartPole")
        self._font = pygame.font.Font(None, 32)
        self._clock = pygame.time.Clock()
        self.current_display_text = ""
        self.main_bg_color = (30, 30, 30)
        self.pole_length = -1 
        self.pole_weight = -1   
        # Color Palette
        self.BG_TOP    = (15,  20,  35)
        self.BG_BOT    = (30,  40,  65)
        self.WHITE     = (240, 245, 255)
        self.YELLOW    = (255, 215,  50)
        self.RED       = (220,  60,  60)
        self.GREEN     = (60,  200, 100)
        self.GREY      = (120, 130, 150)
        self.PANEL_BG  = (0,   0,   0, 160)

        # Thresholds (matching gymnasium defaults)
        self.MAX_ANGLE  = 0.2095
        self.MAX_CART   = 2.4
        self.FPS               = 60
        self.NUDGE_STRENGTH    = 0.5
        self.NUDGE_DISPLAY_TTL = 20 
    def get_input_popup(self, display_message):
        """
        creates a box for user input
        """
        user_text = ""
        is_typing = True
        
        # Defines the pop up box dimensions and positioning
        box_width, box_height = 450, 150
        box_x = (self.screen_width - box_width) // 2
        box_y = (self.screen_height - box_height) // 2
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)

        while is_typing:

            for event in pygame.event.get():
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        is_typing = False  #
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    else:
                        user_text += event.unicode

            # draws text box
            pygame.draw.rect(self._screen, (50, 50, 50), box_rect)
            prompt_surf = self._font.render(display_message, True, (255, 255, 255)) 
            input_surf = self._font.render(user_text + "|", True, (0, 0, 0)) # user text color
            
            self._screen.blit(prompt_surf, (box_rect.x + 20, box_rect.y + 25))
            self._screen.blit(input_surf, (box_rect.x + 20, box_rect.y + 80))


            pygame.display.flip()
            self._clock.tick(30) 

        return user_text


    def lerp_color(self,a, b, t):
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


    def bar_color(self,fraction):
        """Map a normalised fraction to a green → yellow → red colour gradient.

        Args:
            fraction (float): Value in ``[0.0, 1.0]`` representing how close a
                sensor reading is to its danger threshold. ``0.0`` is safe
                (green) and ``1.0`` is critical (red).

        Returns:
            tuple[int, int, int]: RGB colour corresponding to the given fraction.
        """
        if fraction < 0.5:
            return self.lerp_color(self.GREEN, self.YELLOW, fraction * 2)
        return self.lerp_color(self.YELLOW, self.RED, (fraction - 0.5) * 2)


    def draw_gradient_bg(self,surface):
        """Fill a surface with a vertical top-to-bottom gradient background.

        Draws a gradient from ``BG_TOP`` at the top edge to ``BG_BOT`` at the
        bottom edge, one horizontal line at a time.

        Args:
            surface (pygame.Surface): The Pygame surface to draw onto. Modified
                in place.

        Returns:
            None
        """
        for y in range(self.screen_height):
            t = y / self.screen_height
            color = self.lerp_color(self.BG_TOP, self.BG_BOT, t)
            pygame.draw.line(surface, color, (0, y), (self.screen_width, y))


    def draw_status_bar(self, surface, font_sm, label, value, max_val, x, y, w=160, h=14):
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
        color    = self.bar_color(fraction)

        pygame.draw.rect(surface, (40, 50, 70), (x, y, w, h), border_radius=4)
        if fraction > 0:
            pygame.draw.rect(surface, color, (x, y, int(w * fraction), h), border_radius=4)
        pygame.draw.rect(surface, self.GREY, (x, y, w, h), 1, border_radius=4)

        lbl = font_sm.render(f"{label}: {value:+.3f}", True, self.WHITE)
        surface.blit(lbl, (x, y - 18))


    def draw_hud(self,surface, fonts, episode, step, total_reward,
                cart_pos, pole_angle, nudge_ttl, last_nudge_dir):
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
        panel.fill(self.PANEL_BG)
        surface.blit(panel, (pad, pad))

        stats = [
            f"Episode : {episode}",
            f"Step    : {step}",
            f"Reward  : {total_reward:+.2f}",
        ]
        for i, line in enumerate(stats):
            shadow = font_lg.render(line, True, (0, 0, 0))
            text   = font_lg.render(line, True, self.WHITE)
            surface.blit(shadow, (pad + 9, pad + 9 + i * 26))
            surface.blit(text,   (pad + 8, pad + 8 + i * 26))

        # --- sensor bars ---
        bar_x = pad + 8
        bar_y = pad + 100
        self.draw_status_bar(surface, font_sm, "Cart ", cart_pos,  self.MAX_CART,  bar_x, bar_y)
        self.draw_status_bar(surface, font_sm, "Angle", pole_angle, self.MAX_ANGLE, bar_x, bar_y + 42)

        # --- nudge indicator ---
        if nudge_ttl > 0:
            alpha  = int(255 * (nudge_ttl / self.NUDGE_DISPLAY_TTL))
            label  = "◀  NUDGE LEFT" if last_nudge_dir < 0 else "NUDGE RIGHT  ▶"
            nudge_surf = font_lg.render(label, True, self.YELLOW)
            nudge_surf.set_alpha(alpha)
            nx = self.screen_width // 2 - nudge_surf.get_width() // 2
            surface.blit(nudge_surf, (nx, self.screen_height - 44))

        # --- controls legend (bottom-right) ---
        legend = ["← → : nudge pole", "R : reset", "Q / Esc : quit"]
        for i, line in enumerate(legend):
            t = font_sm.render(line, True, self.GREY)
            surface.blit(t, (self.screen_width - t.get_width() - pad,
                            self.screen_height - pad - (len(legend) - i) * 20))
            














    def run_simulation(self,CustomCartPole):
        """
        Runs the actual AI simulation loop. 
        Returns 'RESTART' if 'U' is pressed, 'QUIT' if closed.
        """
        # --- Environment Setup ---
        # Assuming CustomCartPole is accessible here
        # base_env = CustomCartPole() 
        # base_env.length = self.pole_length / 2.0
        # base_env.masspole = self.pole_weight
        # base_env.total_mass = base_env.masspole + base_env.masscart
        # base_env.polemass_length = base_env.masspole * base_env.length
        
        # Placeholder for demonstration if CustomCartPole isn't imported:
        base_env = CustomCartPole()
        base_env.set_parameters(self.pole_lengthlength,self.pole_weightweight)
        env = TimeLimit(base_env, max_episode_steps=500)
        model = PPO.load("ppo_cartpole_yong_4-26", env=env)

        # Pygame setup
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("CartPole — Arrow keys to nudge")
        font_lg = pygame.font.SysFont("monospace", 17, bold=True)
        font_sm = pygame.font.SysFont("monospace", 13)
        clock   = pygame.time.Clock()

        obs, _         = env.reset()
        episode        = 1
        step_count     = 0
        total_reward   = 0.0
        nudge_ttl      = 0
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
                        step_count   = 0
                        total_reward = 0.0

            pole_nudge = 0.0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                pole_nudge     = -NUDGE_STRENGTH
                nudge_ttl      = NUDGE_DISPLAY_TTL
                last_nudge_dir = -1
            elif keys[pygame.K_RIGHT]:
                pole_nudge     = NUDGE_STRENGTH
                nudge_ttl      = NUDGE_DISPLAY_TTL
                last_nudge_dir = 1

            env.unwrapped.state[3] += pole_nudge
            obs = np.array(env.unwrapped.state, dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            step_count   += 1
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
                episode, step_count, total_reward,
                cart_pos, pole_angle,
                nudge_ttl, last_nudge_dir,
            )

            pygame.display.flip()

            if terminated or truncated:
                obs, _       = env.reset()
                episode     += 1
                step_count   = 0
                total_reward = 0.0

            clock.tick(FPS)

        pygame.quit()
        env.close()


    def main(self):
        program_active = True
        while program_active:
            # Length parameter & Validation
            text = "Enter new Pole Length (m):"
            while True:
                result = self.get_input_popup(text)
                try:
                    self.pole_length = float(result)
                    if self.pole_length <= 0: raise ValueError
                    break
                except ValueError:
                    text = "Invalid! Enter Pole Length (m):"

            # Weight Paramter & Validation
            text = "Enter new Weight (kg):"
            while True:
                result = self.get_input_popup(text)
                try:
                    self.pole_weight = float(result)
                    if self.pole_weight <= 0: raise ValueError
                    break
                except ValueError:
                    text = "Invalid! Enter Weight (kg):"



            # Runs the simulation
            signal = self.run_simulation()
            
            if signal == "QUIT":
                program_active = False
            elif signal == "RESTART":
                continue # Loops back to the input phase

        pygame.quit()


# test
if __name__ == "__main__":
    my_model = view()
    my_model.main()