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
        self.screen_size = {
            "screen_width": 800,
            "screen_height": 800,
        }
        self._screen = pygame.display.set_mode(
            (self.screen_size["screen_width"], self.screen_size["screen_height"])
        )
        pygame.display.set_caption("CartPole")
        self._font = pygame.font.Font(None, 32)
        self._clock = pygame.time.Clock()
        self.current_display_text = ""
        # Color Palette
        self.color_palette = {
            "bg_top": (15, 20, 35),
            "bg_bot": (30, 40, 65),
            "white": (240, 245, 255),
            "yellow": (255, 215, 50),
            "red": (220, 60, 60),
            "green": (60, 200, 100),
            "grey": (120, 130, 150),
            "panel_bg": (0, 0, 0, 160),
            "main_bg_color": (30, 30, 30),
        }

        # Thresholds (matching gymnasium defaults)
        self.thresholds = {
            "max_angle": 0.2095,
            "max_cart": 2.4,
            "fps": 60,
            "nudge_strength": 0.5,
            "nudge_display_ttl": 20,
        }

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
        box_x = (self.screen_size["screen_width"] - box_width) // 2
        box_y = (self.screen_size["screen_height"] - box_height) // 2
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
            return self.lerp_color(
                self.color_palette["green"], self.color_palette["yellow"], fraction * 2
            )
        return self.lerp_color(
            self.color_palette["yellow"],
            self.color_palette["red"],
            (fraction - 0.5) * 2,
        )

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
        for y in range(self.screen_size["screen_height"]):
            t = y / self.screen_size["screen_height"]
            color = self.lerp_color(
                self.color_palette["bg_top"], self.color_palette["bg_bot"], t
            )
            pygame.draw.line(
                surface, color, (0, y), (self.screen_size["screen_width"], y)
            )

    def draw_status_bar(self, surface, font_sm, value, config):
        """Draw a labelled horizontal progress bar at the specified position.

        Args:
            surface (pygame.Surface): The Pygame surface to draw onto.
            font_sm (pygame.font.Font): Font for the label.
            value (float): Current reading.
            config (dict): Contains 'label', 'max', 'x', and 'y'.
        """
        fraction = min(abs(value) / config["max"], 1.0)
        color = self.bar_color(fraction)

        pygame.draw.rect(
            surface, (40, 50, 70), (config["x"], config["y"], 160, 14), border_radius=4
        )
        if fraction > 0:
            pygame.draw.rect(
                surface,
                color,
                (config["x"], config["y"], int(160 * fraction), 14),
                border_radius=4,
            )

        lbl = font_sm.render(
            f"{config['label']}: {value:+.3f}", True, self.color_palette["white"]
        )
        surface.blit(lbl, (config["x"], config["y"] - 18))

    def draw_hud(self, surface, fonts, hud_data):
        """Draw the full HUD overlay onto the given surface.

        Args:
            surface (pygame.Surface): The Pygame surface to draw onto.
            fonts (tuple): A 2-tuple of (font_lg, font_sm).
            hud_data (dict): Dictionary containing all simulation metrics.
        """
        # Draw Stats Panel
        panel = pygame.Surface((200, 160), pygame.SRCALPHA)  # pylint: disable=no-member
        panel.fill(self.color_palette["panel_bg"])
        surface.blit(panel, (14, 14))

        for i, line in enumerate(
            [
                f"Episode : {hud_data['episode']}",
                f"Step    : {hud_data['step']}",
                f"Reward  : {hud_data['reward']:+.2f}",
            ]
        ):
            surface.blit(
                fonts[0].render(line, True, self.color_palette["white"]),
                (22, 22 + i * 26),
            )

        # Draw Sensor Bars
        self.draw_status_bar(
            surface,
            fonts[1],
            hud_data["cart_pos"],
            {"label": "Cart ", "max": self.thresholds["max_cart"], "x": 22, "y": 114},
        )
        self.draw_status_bar(
            surface,
            fonts[1],
            hud_data["pole_angle"],
            {"label": "Angle", "max": self.thresholds["max_angle"], "x": 22, "y": 156},
        )

        # Draw Nudge & Legend
        self._draw_hud_extras(surface, fonts, hud_data)

    def _draw_hud_extras(self, surface, fonts, hud_data):
        """Helper to draw the nudge indicator and controls legend."""
        if hud_data["nudge_ttl"] > 0:
            alpha = int(
                255 * (hud_data["nudge_ttl"] / self.thresholds["nudge_display_ttl"])
            )
            label = (
                "◀  NUDGE LEFT" if hud_data["last_nudge_dir"] < 0 else "NUDGE RIGHT  ▶"
            )
            nudge_surf = fonts[0].render(label, True, self.color_palette["yellow"])
            nudge_surf.set_alpha(alpha)
            surface.blit(
                nudge_surf,
                (
                    self.screen_size["screen_width"] // 2 - nudge_surf.get_width() // 2,
                    self.screen_size["screen_height"] - 44,
                ),
            )

        for i, line in enumerate(["<- | -> : nudge pole", "R : reset", "Esc : quit"]):
            t = fonts[1].render(line, True, self.color_palette["grey"])
            surface.blit(
                t,
                (
                    self.screen_size["screen_width"] - t.get_width() - 14,
                    self.screen_size["screen_height"] - 14 - (3 - i) * 20,
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
        """Provides public access to the main clock."""
        return self._clock

    @property
    def fps(self):
        """Provides public access to the main fps."""
        return self.thresholds["fps"]
