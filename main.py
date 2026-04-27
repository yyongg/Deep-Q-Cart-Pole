"""
This module serves as the main execution script for the Deep-Q Cart-Pole
simulation.
"""

import pygame
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from view import View
from controller import CartPoleController

# --- THE BARE MINIMUM CALLBACK ---
# Pushes the training frames to your View window so Pygame doesn't crash
class LiveTraining(BaseCallback):
    def __init__(self, view, env):
        super().__init__()
        self.view = view
        self.env = env

    def _on_step(self):
        pygame.event.pump() # Stops macOS/Windows from thinking the app froze
        
        # Render every 3rd step so it trains fast but remains visible
        if self.num_timesteps % 3 == 0: 
            frame = np.transpose(self.env.render(), (1, 0, 2))
            surf = pygame.transform.scale(pygame.surfarray.make_surface(frame), (800, 800))
            self.view.screen.blit(surf, (0, 0))
            
            # Optional: Draw a red 'TRAINING' text so you know it's working
            text = self.view.font.render(f"TRAINING... Step {self.num_timesteps}", True, (255, 50, 50))
            self.view.screen.blit(text, (20, 20))
            
            pygame.display.flip()
        return True
# ---------------------------------

def main():
    pygame.init() 
    view = View()
    controller = CartPoleController(view)

    program_running = True
    while program_running:
        
        # 1. Ask for inputs via your View
        length, weight = controller.get_user_inputs()
        
        # 2. Setup the model (Returns True if no .zip file is found)
        needs_training = controller.setup_env_and_model(length, weight)

        # 3. Watch it train!
        if needs_training:
            print("Live training starting...")
            cb = LiveTraining(view, controller.env)
            
            # NOTE: I lowered this to 100,000 so you aren't waiting forever right now.
            controller.model.learn(total_timesteps=200_000, callback=cb)
            controller.model.save("dqn_swing_up")

        # 4. Interactive Mode
        result = controller.run_simulation()
        
        if result == "QUIT":
            program_running = False

    pygame.quit() 

if __name__ == "__main__":
    main()