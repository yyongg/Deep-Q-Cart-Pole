from view import view
from controller import CartPoleController
import pygame

def main():
    # Initialize the View (The Windows and Fonts)
    app_view = view()
    
    # Initialize the Controller and give it the View
    app_controller = CartPoleController(app_view)
    
    program_running = True
    while program_running:
        # Step 1: Setup parameters (Inputs)
        app_controller.setup_session()
        
        # Step 2: Run the simulation
        # This will return "RESTART" if you press 'U', or "QUIT" if you close
        result = app_controller.run_simulation()
        
        if result == "QUIT":
            program_running = False
            
    pygame.quit()

if __name__ == "__main__":
    main()