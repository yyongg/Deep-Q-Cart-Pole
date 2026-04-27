import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
from controller import CartPoleController

@pytest.fixture
def controller():
    view = MagicMock()
    view.thresholds = {
        "nudge_strength": 0.5,
        "nudge_display_ttl": 20
    }
    
    c = CartPoleController(view)
    
    # Mock Environment with correct return values for unpacking
    c.env = MagicMock()
    c.env.unwrapped.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    c.env.reset.return_value = (np.zeros(5), {})
    c.env.step.return_value = (np.zeros(5), 0.0, False, False, {})
    
    # Mock Model with correct return values for unpacking
    c.model = MagicMock()
    c.model.predict.return_value = (0, None)
    
    return c

def test_nudge_right_updates_state(controller):
    """
    Verifies that a right-nudge input correctly increases the pole's angular 
    velocity, ensuring directional symmetry in the physics manipulation.
    """
    mock_keys = [0] * 512
    with patch("pygame.key.get_pressed", return_value=mock_keys), \
         patch("pygame.K_LEFT", 0), \
         patch("pygame.K_RIGHT", 1):
        
        mock_keys[1] = 1
        controller.nudges(nudge_ttl=0, last_nudge_dir=0)
        
        assert controller.env.unwrapped.state[3] == 0.5

def test_nudge_returns_updated_ui_metadata(controller):
    """
    Confirms that the nudge function returns the correct TTL (Time To Live) 
    and direction flag, which are critical for the View to render the visual 
    nudge indicator.
    """
    mock_keys = [0] * 512
    with patch("pygame.key.get_pressed", return_value=mock_keys), \
         patch("pygame.K_LEFT", 0), \
         patch("pygame.K_RIGHT", 1):
        
        mock_keys[0] = 1
        _, ttl, direction = controller.nudges(nudge_ttl=0, last_nudge_dir=0)
        
        assert ttl == 20
        assert direction == -1

def test_setup_logic_creates_new_model_if_missing(controller):
    """
    Tests the decision-making logic of the environment setup to ensure a 
    fresh DQN model is initialized with specific hyperparameters when no 
    pre-trained weight file is detected.
    """
    with patch("controller.CustomCartPole"), \
         patch("controller.TimeLimit"), \
         patch("os.path.exists", return_value=False), \
         patch("controller.DQN") as mock_dqn:
        
        needs_train = controller.setup_env_and_model(1.0, 0.1)
        
        assert needs_train is True
        mock_dqn.assert_called_once()

def test_setup_logic_loads_existing_model(controller):
    """
    Ensures that if a saved model exists, the controller bypasses training 
    and loads the weights into the current environment to save time and 
    resources.
    """
    with patch("controller.CustomCartPole"), \
         patch("controller.TimeLimit"), \
         patch("os.path.exists", return_value=True), \
         patch("controller.DQN.load") as mock_load:
        
        needs_train = controller.setup_env_and_model(1.0, 0.1)
        
        assert needs_train is False
        mock_load.assert_called_once()

def test_observation_trig_conversion_accuracy(controller):
    """
    Validates the mathematical conversion of a 90-degree pole angle into 
    sine and cosine components, ensuring the DQN receives accurate spatial 
    data regardless of the pole's orientation.
    """
    controller.env.unwrapped.state = np.array([0.0, 0.0, np.pi/2, 0.0], dtype=np.float32)
    
    mock_keys = [0] * 512
    with patch("pygame.key.get_pressed", return_value=mock_keys), \
         patch("pygame.K_LEFT", 0), \
         patch("pygame.K_RIGHT", 1):
        
        controller.nudges(0, 0)
        
        args, _ = controller.model.predict.call_args
        obs = args[0]
        
        assert pytest.approx(obs[2], abs=1e-5) == 0.0
        assert pytest.approx(obs[3], abs=1e-5) == 1.0

def test_simulation_event_handling_restart(controller):
    """
    Checks the event polling logic in the main simulation loop to verify 
    that the 'R' key correctly triggers a 'RESTART' signal to the main 
    execution script.
    """
    mock_event = MagicMock()
    mock_event.type = 768 # pygame.KEYDOWN
    mock_event.key = 114  # pygame.K_r
    
    with patch("pygame.event.get", return_value=[mock_event]), \
         patch("pygame.key.get_pressed", return_value=[0]*512), \
         patch("controller.pygame.K_r", 114), \
         patch("controller.pygame.surfarray.make_surface"), \
         patch("controller.pygame.transform.scale"), \
         patch("controller.pygame.display.flip"):
        
        result = controller.run_simulation()
        
        assert result == "RESTART"

def test_simulation_event_handling_quit(controller):
    """
    Ensures that the application's termination signal (Escape key) is 
    correctly captured and processed, preventing the simulation from 
    hanging or failing to close properly.
    """
    mock_event = MagicMock()
    mock_event.type = 768 # pygame.KEYDOWN
    mock_event.key = 27   # pygame.K_ESCAPE
    
    with patch("pygame.event.get", return_value=[mock_event]), \
         patch("pygame.key.get_pressed", return_value=[0]*512), \
         patch("controller.pygame.K_ESCAPE", 27), \
         patch("controller.pygame.surfarray.make_surface"), \
         patch("controller.pygame.transform.scale"), \
         patch("controller.pygame.display.flip"):
        
        result = controller.run_simulation()
        
        assert result == "QUIT"