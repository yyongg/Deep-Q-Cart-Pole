"""
MVC - View

This module handles the 'View' component of the Model-View-Controller (MVC) 
architecture for a Cart-Pole simulation. It manages the Pygame 
window, renders the simulation, and shows a HUD for metrics for the user nudging feature.
It also handles user input pop-ups for setting parameters.
"""

import pygame


class View:
    """
    Handles the rendering and user interface for the Cart-Pole simulation.

    This class manages the Pygame window, draws the environment
    and provides utility functions for color interpolation 
    and UI elements like status bars and input pop-ups.

    Attributes:
        screen_width (int): Width of the display window in pixels.
        screen_height (int): Height of the display window in pixels.
        max_angle (float): The failure threshold for the pole angle (radians).
        max_cart (float): The failure threshold for the cart position (meters).
        fps (int): Frames per second limit for the simulation.
    """

    def __init__(self):
        """Initializes the Pygame environment, display settings, and color palette."""
        # pygame window variable setup
        pygame.init()  # pylint: disable=no-member
        self.screen_width = 800
        self.screen_height = 800
        self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("CartPole")
        self._font = pygame.font.Font(None, 32)
        self._clock = pygame.time.Clock()
        self.current_display_text = ""
        self.main_bg_color = (30, 30, 30)
        #default parameters for pole length and weight
        self.pole_length = -1
        self.pole_weight = -1
        # Color Palette
        self.bg_top = (15, 20, 35)
        self.bg_bot = (30, 40, 65)
        self.white = (240, 245, 255)
        self.yellow = (255, 215, 50)
        self.red = (220, 60, 60)
        self.green = (60, 200, 100)
        self.grey = (120, 130, 150)
        self.panel_bg = (0, 0, 0, 160)

        # Thresholds (matching gymnasium defaults)
        self.max_angle = 0.2095
        self.max_cart = 2.4
        self.fps = 60
        self.nudge_strength = 0.5
        self.nudge_display_ttl = 20

    def get_input_popup(self, display_message):
        """
        Displays a graphical input box and captures keyboard input from the user.

        Args:
            display_message (str): The prompt text shown above the input field.

        Returns:
            user_text (int): The value given by the user
        """
        user_text = ""
        is_typing = True

        # Defines the pop up box dimensions and positioning
        box_width, box_height = 450, 150
        box_x = (self.screen_width - box_width) // 2
        box_y = (self.screen_height - box_height) // 2
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)

        # loop for taking user input
        while is_typing:

            for event in pygame.event.get():

                if event.type == pygame.KEYDOWN:  # pylint: disable=no-member
                    if event.key == pygame.K_RETURN:  # pylint: disable=no-member
                        is_typing = False  #
                    elif event.key == pygame.K_BACKSPACE:  # pylint: disable=no-member
                        user_text = user_text[:-1]
                    else:
                        user_text += event.unicode

            # draws text box
            pygame.draw.rect(self._screen, (50, 50, 50), box_rect)
            prompt_surf = self._font.render(display_message, True, (255, 255, 255))
            input_surf = self._font.render(
                user_text + "|", True, (0, 0, 0)
            )  # user text color

            self._screen.blit(prompt_surf, (box_rect.x + 20, box_rect.y + 25))
            self._screen.blit(input_surf, (box_rect.x + 20, box_rect.y + 80))

            pygame.display.flip()
            self._clock.tick(30)

        return user_text

    def lerp_color(self, a, b, t):
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

    def bar_color(self, fraction):
        """Map a normalised fraction to a green → yellow → red colour gradient.

        Args:
            fraction (float): Value in ``[0.0, 1.0]`` representing how close a
                sensor reading is to its danger threshold. ``0.0`` is safe
                (green) and ``1.0`` is critical (red).

        Returns:
            tuple[int, int, int]: RGB colour corresponding to the given fraction.
        """
        if fraction < 0.5:
            return self.lerp_color(self.green, self.yellow, fraction * 2)
        return self.lerp_color(self.yellow, self.red, (fraction - 0.5) * 2)

    def draw_gradient_bg(self, surface):
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
            color = self.lerp_color(self.bg_top, self.bg_bot, t)
            pygame.draw.line(surface, color, (0, y), (self.screen_width, y))

    def draw_status_bar(
        self, surface, font_sm, label, value, max_val, x, y, w=160, h=14):

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
        color = self.bar_color(fraction)

        pygame.draw.rect(surface, (40, 50, 70), (x, y, w, h), border_radius=4)
        if fraction > 0:
            pygame.draw.rect(
                surface, color, (x, y, int(w * fraction), h), border_radius=4
            )
        pygame.draw.rect(surface, self.grey, (x, y, w, h), 1, border_radius=4)

        lbl = font_sm.render(f"{label}: {value:+.3f}", True, self.white)
        surface.blit(lbl, (x, y - 18))

    def draw_hud(
        self,
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
        panel = pygame.Surface((200, 160), pygame.SRCALPHA)  # pylint: disable=no-member
        panel.fill(self.panel_bg)
        surface.blit(panel, (pad, pad))

        stats = [
            f"Episode : {episode}",
            f"Step    : {step}",
            f"Reward  : {total_reward:+.2f}",
        ]
        for i, line in enumerate(stats):
            shadow = font_lg.render(line, True, (0, 0, 0))
            text = font_lg.render(line, True, self.white)
            surface.blit(shadow, (pad + 9, pad + 9 + i * 26))
            surface.blit(text, (pad + 8, pad + 8 + i * 26))

        # --- sensor bars ---
        bar_x = pad + 8
        bar_y = pad + 100
        self.draw_status_bar(
            surface, font_sm, "Cart ", cart_pos, self.max_cart, bar_x, bar_y
        )
        self.draw_status_bar(
            surface, font_sm, "Angle", pole_angle, self.max_angle, bar_x, bar_y + 42
        )

        # --- nudge indicator ---
        if nudge_ttl > 0:
            alpha = int(255 * (nudge_ttl / self.nudge_display_ttl))
            label = "◀  NUDGE LEFT" if last_nudge_dir < 0 else "NUDGE RIGHT  ▶"
            nudge_surf = font_lg.render(label, True, self.yellow)
            nudge_surf.set_alpha(alpha)
            nx = self.screen_width // 2 - nudge_surf.get_width() // 2
            surface.blit(nudge_surf, (nx, self.screen_height - 44))

        # --- controls legend (bottom-right) ---
        legend = ["<- | -> : nudge pole", "R : reset", "Esc : quit"]
        for i, line in enumerate(legend):
            t = font_sm.render(line, True, self.grey)
            surface.blit(
                t,
                (
                    self.screen_width - t.get_width() - pad,
                    self.screen_height - pad - (len(legend) - i) * 20,
                ),
            )

    @property
    def screen(self):
        """Provides public access to the display surface."""
        return self._screen

    @property
    def font(self):
        """Provides public access to the main font."""
        return self._font
    
    @property
    def clock(self):
        """Provides public access to the main font."""
        return self._clock