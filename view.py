import pygame
import sys

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

    def run_main_game(self):
        """
        Runs the main game window
        """
        running = True
        # calls for initial parameter setup

        # length parameter
        text = "Enter new Pole Length:"
        while True:
            result = self.get_input_popup(text)
            # ensures input is valid
            try:
                self.pole_length = float(result)
                if self.pole_length < 0:
                    raise ValueError
                break
            except ValueError:
                text = "Invalid input, Enter new Pole Length:"
                




        #Weight paramter
        text = "Enter new Weight:"
        while True:
            result = self.get_input_popup(text)
            try:
                self.pole_weight = float(result)
                if self.pole_weight < 0:
                    raise ValueError
                break
            except ValueError:
                text = "Invalid Input, Enter new weight:"







        # EVERYTHING BELOW IS EXTRA/POSSIBLY NOT NEEDED IN FINAL PROGRAM




        while running:
            self._screen.fill(self.main_bg_color)

          
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                    # Below is code to reupdate the variables
            """
                if event.type == pygame.KEYDOWN:


                    # Length parameter
                    if event.key == pygame.K_l:
                        result = self.get_input_popup("Enter New Pole Length(m):")
                        # ensures input is valid
                        try:
                            self.pole_length = float(result)
                            if self.pole_length < 0:
                                raise ValueError
                        except ValueError:
                            self.current_display_text = "Invalid Input"



                    # Weight paramter
                    if event.key == pygame.K_p:
                        result = self.get_input_popup("Enter Weight(kg):")
                        try:
                            self.pole_weight = float(result)
                            if self.pole_weight < 0:
                                raise ValueError
                        except ValueError:
                            self.current_display_text = "Invalid Input"
            """
            # Simulation viewing goes here?
            info_surf = self._font.render(self.current_display_text, True, (200, 200, 200))
            self._screen.blit(info_surf, (50, 350))
            
            # Display inputted parameters
            stats_str = f"Length(m): {self.pole_length} | Weight(kg): {self.pole_weight}"
            stats_surf = self._font.render(stats_str, True, (100, 150, 255))
            self._screen.blit(stats_surf, (50, 400))

            # Update main display frame
            pygame.display.flip()
            self._clock.tick(60) # Pacing 

        pygame.quit()

# test
if __name__ == "__main__":
    my_model = view()
    my_model.run_main_game()