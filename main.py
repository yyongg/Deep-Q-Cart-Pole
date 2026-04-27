"""
This module serves as the main execution script for the Deep-Q Cart-Pole
simulation.
"""

import pygame
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from view import View
from controller import CartPoleController


class LiveTraining(BaseCallback):
    """
    A reinforcement learning callback designed to synchronize the agent's 
    training progress with a Pygame display. It ensures the application 
    remains responsive to the operating system while providing real-time 
    visual feedback of the learning process.
    """
    def __init__(self, view, env):
        super().__init__()
        self.view = view
        self.env = env

    def _on_step(self):
        """
        Executes logic at every training step to pump the Pygame event queue 
        and conditionally render the environment frame. This prevents the 
        window from being flagged as unresponsive and displays the agent's 
        current swing-up attempts at a set frequency.
        """
        pygame.event.pump()
        
        if self.num_timesteps % 3 == 0: 
            frame = np.transpose(self.env.render(), (1, 0, 2))
            surf = pygame.transform.scale(pygame.surfarray.make_surface(frame), (800, 800))
            self.view.screen.blit(surf, (0, 0))
            
            text = self.view.font.render(f"TRAINING... Step {self.num_timesteps}", True, (255, 50, 50))
            self.view.screen.blit(text, (20, 20))
            
            pygame.display.flip()
        return True


def main():
    """
    The entry point for the application. It orchestrates the flow of the 
    simulation by initializing the MVC components, collecting user-defined 
    physics parameters, and managing the transition between the automated 
    training phase and the interactive user-controlled simulation.
    """
    pygame.init() 
    view = View()
    controller = CartPoleController(view)

    program_running = True
    while program_running:
        """
        The master execution loop that remains active until the user signals 
        to quit. It facilitates restarts, allowing for new physics 
        parameters to be set and tested dynamically.
        """
        length, weight = controller.get_user_inputs()
        
        needs_training = controller.setup_env_and_model(length, weight)

        if needs_training:
            """
            Triggers the training sequence if a pre-trained model is not detected. 
            The LiveTraining callback is utilized to maintain the visual output 
            during the intensive mathematical calculation phase.
            """
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