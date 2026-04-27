"""
This module serves as the main execution script for the Deep-Q Cart-Pole
simulation. It handles the orchestration of the training and evaluation
phases, ensuring the agent learns to balance the pole before the
interactive simulation begins.
"""

import pygame
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from view import View
from controller import CartPoleController

# pylint: disable=no-member


class LiveTraining(BaseCallback):
    """
    A callback class used during the training phase to provide real-time
    visual updates. It renders the state of the environment to the Pygame
    window periodically, allowing for observation of the learning process
    without blocking the event loop.
    """

    def __init__(self, view, env):
        """
        Initializes the callback with references to the display View and
        the active environment. Sets the training context for visual feedback.
        """
        super().__init__()
        self.view = view
        self.env = env

    def _on_step(self) -> bool:
        """
        Processes a single training step. Ponds the event queue to keep the
        window responsive and renders the environment frame to the screen
        every three timesteps to maintain simulation performance.
        """
        pygame.event.pump()

        if self.num_timesteps % 3 == 0:
            frame = np.transpose(self.env.render(), (1, 0, 2))
            surf = pygame.transform.scale(
                pygame.surfarray.make_surface(frame), (800, 800)
            )
            self.view.screen.blit(surf, (0, 0))

            text = self.view.font.render(
                f"TRAINING... Step {self.num_timesteps}", True, (255, 50, 50)
            )
            self.view.screen.blit(text, (20, 20))

            pygame.display.flip()
        return True


def main():
    """
    The main application entry point. It initializes the Pygame context,
    sets up the MVC components, and manages the primary execution loop
    which alternates between user parameter input, agent training, and
    the interactive simulation mode.
    """
    pygame.init()
    view = View()
    controller = CartPoleController(view)

    program_running = True
    while program_running:
        length, weight = controller.get_user_inputs()

        needs_training = controller.setup_env_and_model(length, weight)

        if needs_training:
            print("Live training starting...")
            cb = LiveTraining(view, controller.env)

            controller.model.learn(total_timesteps=200_000, callback=cb)
            controller.model.save("dqn_swing_up")

        result = controller.run_simulation()

        if result == "QUIT":
            program_running = False

    pygame.quit()


if __name__ == "__main__":
    main()
