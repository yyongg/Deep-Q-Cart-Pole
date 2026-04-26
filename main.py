"""
This module serves as the main execution script for the Deep-Q Cart-Pole
simulation.
"""

import pygame
from view import View
from controller import CartPoleController


def main():
    """
    Initializes the View and Controller, then enters a loop that allows
    the user to configure parameters and run the simulation multiple times
    until the application is closed.

    Args:
        None
    Returns:
        None

    """
    # Initialize each class
    view = View()
    controller = CartPoleController(view)

    program_running = True
    while program_running:
        # set up parameters from user
        controller.setup_session()

        # Runs the simulation
        result = controller.run_simulation()

        if result == "QUIT":
            program_running = False

    pygame.quit()  # pylint: disable=no-member


if __name__ == "__main__":
    main()  # Executes code
