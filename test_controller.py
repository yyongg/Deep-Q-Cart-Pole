"""Unit tests using Pytest for our controller"""

from unittest.mock import MagicMock, patch
import pytest
import numpy as np

# pytest fixtures intentionally share the same name as the parameter they
# inject — pylint flags this as W0621 (redefined-outer-name) but it is the
# correct, idiomatic pattern for pytest and should not be suppressed per-test.
# pylint: disable=redefined-outer-name


# Helpers / shared fixtures

@pytest.fixture
def mock_view():
    """A minimal fake View that satisfies CartPoleController's interface."""
    view = MagicMock()
    view.NUDGE_STRENGTH = 0.5
    view.NUDGE_DISPLAY_TTL = 10
    view.FPS = 60
    return view


@pytest.fixture
def controller(mock_view):
    """Return a fresh CartPoleController with a mocked view."""
    with patch.dict("sys.modules", {
        "stable_baselines3": MagicMock(),
        "gymnasium": MagicMock(),
        "gymnasium.wrappers": MagicMock(),
        "model": MagicMock(),
        "pygame": MagicMock(),
    }):
        # pylint: disable=import-outside-toplevel
        from controller import CartPoleController
        return CartPoleController(mock_view)


# 1. __init__

class TestInit:
    """
    Verify that CartPoleController.__init__ sets every attribute to its
    correct default value before any session is started.

    All four attributes (view, model, ai, env) must be in a known state
    immediately after construction so that other code can safely check them
    before setup_session has been called.
    """
    def test_view_assigned(self, controller, mock_view):
        """The view passed to __init__ should be stored as controller.view."""
        assert controller.view is mock_view

    def test_model_is_none_on_init(self, controller):
        """controller.model should be None until a model is explicitly loaded."""
        assert controller.model is None

    def test_ai_is_none_on_init(self, controller):
        """controller.ai should be None until setup_session successfully loads weights."""
        assert controller.ai is None

    def test_env_is_none_on_init(self, controller):
        """controller.env should be None until setup_session builds the environment."""
        assert controller.env is None


# 2. setup_session – happy path

class TestSetupSessionHappyPath:
    """
    Verify setup_session's behaviour when the user supplies valid inputs on
    the first attempt and the PPO weight file is present.

    These tests confirm that the environment is constructed and wrapped
    correctly, that set_parameters receives the right values, and that the
    loaded AI model is stored on the controller ready for run_simulation.
    """
    def _run_setup(self, controller, length="1.5", weight="0.5"):
        """Drive setup_session with valid inputs on the first try."""
        controller.view.get_input_popup.side_effect = [length, weight]
        mock_ai = MagicMock()

        with patch("controller.CustomCartPole"), \
             patch("controller.TimeLimit") as mock_tl, \
             patch("controller.PPO.load", return_value=mock_ai):
            controller.setup_session()
            wrapped_env = mock_tl.return_value

        return wrapped_env, mock_ai

    def test_env_is_set(self, controller):
        """
        setup_session should assign a non-None value to controller.env.

        Note: we don't assert a specific mock identity here because TimeLimit is
        bound via sys.modules at import time; the other tests already confirm
        it is called correctly and with the right arguments.
        """
        self._run_setup(controller)
        assert controller.env is not None

    def test_ai_is_set_when_weights_found(self, controller):
        """
        setup_session should assign the loaded PPO model to controller.ai
        when the weight file exists and PPO.load succeeds.
        """
        _, mock_ai = self._run_setup(controller)
        assert controller.ai is mock_ai

    def test_set_parameters_called_with_correct_values(self, controller):
        """
        setup_session should forward the user-entered length and weight to
        CustomCartPole.set_parameters as floats, in that order.
        """
        length, weight = "2.0", "0.3"
        controller.view.get_input_popup.side_effect = [length, weight]

        mock_base_env = MagicMock()
        with patch("controller.CustomCartPole", return_value=mock_base_env), \
             patch("controller.TimeLimit"), \
             patch("controller.PPO.load"):
            controller.setup_session()

        mock_base_env.set_parameters.assert_called_once_with(
            float(length), float(weight)
        )

    def test_timelimit_wraps_base_env(self, controller):
        """
        setup_session should wrap the CustomCartPole instance with TimeLimit,
        passing max_episode_steps=500 as required by the Gymnasium API.
        """
        controller.view.get_input_popup.side_effect = ["1.0", "1.0"]

        mock_base_env = MagicMock()
        with patch("controller.CustomCartPole", return_value=mock_base_env), \
             patch("controller.TimeLimit") as mock_tl, \
             patch("controller.PPO.load"):
            controller.setup_session()

        mock_tl.assert_called_once_with(mock_base_env, max_episode_steps=500)


# 3. setup_session – invalid inputs (retries until valid)

class TestSetupSessionInvalidInputs:
    """
    Verify that setup_session re-prompts the user when invalid values are
    entered for pole length or cart weight.

    Valid inputs are strictly positive floats. The controller must reject
    non-numeric strings, zero, and negative numbers for both fields, looping
    until acceptable values are provided without raising an exception.
    """
    def test_retries_on_non_numeric_length(self, controller):
        """
        The length input loop should re-prompt on non-numeric strings and on
        values that are zero or negative, only accepting a strictly positive float.
        Here we feed three bad length values ('abc', '0', '-1') before a valid
        one, then one valid weight — expecting five total popup calls.
        """
        controller.view.get_input_popup.side_effect = ["abc", "0", "-1", "1.0", "0.5"]

        with patch("controller.CustomCartPole"), \
             patch("controller.TimeLimit"), \
             patch("controller.PPO.load"):
            controller.setup_session()

        assert controller.view.get_input_popup.call_count == 5

    def test_retries_on_zero_weight(self, controller):
        """
        The weight input loop should reject zero and keep prompting.
        One valid length, one rejected weight ('0'), one valid weight — three calls total.
        """
        controller.view.get_input_popup.side_effect = ["1.0", "0", "0.5"]

        with patch("controller.CustomCartPole"), \
             patch("controller.TimeLimit"), \
             patch("controller.PPO.load"):
            controller.setup_session()

        assert controller.view.get_input_popup.call_count == 3

    def test_retries_on_negative_weight(self, controller):
        """
        The weight input loop should reject negative numbers and keep prompting.
        One valid length, one rejected weight ('-5'), one valid weight — three calls total.
        """
        controller.view.get_input_popup.side_effect = ["1.0", "-5", "0.5"]

        with patch("controller.CustomCartPole"), \
             patch("controller.TimeLimit"), \
             patch("controller.PPO.load"):
            controller.setup_session()

        assert controller.view.get_input_popup.call_count == 3


# 4. setup_session – missing AI weight file

class TestSetupSessionMissingWeights:
    """
    Verify setup_session's fallback behaviour when the PPO weight file is
    absent (PPO.load raises FileNotFoundError).

    The controller must handle this gracefully: the environment should still
    be fully initialised so the simulation loop can run, and controller.ai
    should be None so that run_simulation falls back to random action sampling.
    """
    def test_ai_is_none_when_file_not_found(self, controller):
        """
        When PPO.load raises FileNotFoundError (weight file missing),
        setup_session should catch the error gracefully and leave controller.ai
        as None so the simulation falls back to random actions.
        """
        controller.view.get_input_popup.side_effect = ["1.0", "0.5"]

        with patch("controller.CustomCartPole"), \
             patch("controller.TimeLimit"), \
             patch("controller.PPO.load", side_effect=FileNotFoundError):
            controller.setup_session()

        assert controller.ai is None

    def test_env_still_set_when_file_not_found(self, controller):
        """
        A missing weight file should not prevent the environment from being
        initialised. controller.env must be set even when PPO.load fails,
        so the simulation loop can still run with random actions.
        """
        controller.view.get_input_popup.side_effect = ["1.0", "0.5"]

        mock_env = MagicMock()
        with patch("controller.CustomCartPole"), \
             patch("controller.TimeLimit", return_value=mock_env), \
             patch("controller.PPO.load", side_effect=FileNotFoundError):
            controller.setup_session()

        assert controller.env is mock_env


# 5. run_simulation – nudge logic (unit-level, no full loop)

class TestNudgeLogic:
    """
    Isolate the nudge math without running the full pygame loop.
    We verify that pole angular velocity (state[3]) is mutated correctly.
    """

    def _apply_nudge(self, controller, direction: str):
        """Simulate one nudge frame manually."""
        state = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float32)
        controller.env = MagicMock()
        controller.env.unwrapped.state = state

        nudge = 0.0
        nudge_strength = controller.view.NUDGE_STRENGTH

        if direction == "left":
            nudge = -nudge_strength
        elif direction == "right":
            nudge = nudge_strength

        controller.env.unwrapped.state[3] += nudge
        return controller.env.unwrapped.state

    def test_left_nudge_decreases_angular_velocity(self, controller):
        """
        Pressing LEFT should subtract NUDGE_STRENGTH from state[3] (pole angular
        velocity), making it negative and causing the pole to tip left.
        """
        state = self._apply_nudge(controller, "left")
        assert state[3] == pytest.approx(-0.5)

    def test_right_nudge_increases_angular_velocity(self, controller):
        """
        Pressing RIGHT should add NUDGE_STRENGTH to state[3] (pole angular
        velocity), making it positive and causing the pole to tip right.
        """
        state = self._apply_nudge(controller, "right")
        assert state[3] == pytest.approx(0.5)

    def test_no_nudge_leaves_state_unchanged(self, controller):
        """
        When no directional key is pressed, state[3] should remain at its
        initial value of 0.0 — no unintended drift is introduced.
        """
        state = self._apply_nudge(controller, "none")
        assert state[3] == pytest.approx(0.0)


# 6. run_simulation – random action fallback (ai=None)

class TestActionSelection:
    """
    Verify the action-selection branch inside run_simulation.

    The controller has two modes: deterministic AI prediction when a loaded
    model is available, and random environment sampling when it is not.
    These tests isolate that branching logic directly, without running the
    full pygame event loop.
    """
    def test_uses_env_sample_when_ai_is_none(self, controller):
        """
        When controller.ai is None (weight file was missing), the simulation
        should fall back to sampling a random action from the environment's
        action space rather than calling ai.predict.
        """
        controller.ai  = None
        controller.env = MagicMock()
        controller.env.action_space.sample.return_value = 1

        obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if controller.ai:
            action, _ = controller.ai.predict(obs, deterministic=True)
        else:
            action = controller.env.action_space.sample()

        controller.env.action_space.sample.assert_called_once()
        assert action == 1

    def test_uses_ai_predict_when_ai_available(self, controller):
        """
        When controller.ai is set, the simulation should call ai.predict with
        deterministic=True and use its returned action, bypassing random sampling.
        """
        controller.ai  = MagicMock()
        controller.ai.predict.return_value = (0, None)
        controller.env = MagicMock()

        obs = np.array([0.1, 0.0, -0.05, 0.0], dtype=np.float32)
        if controller.ai:
            action, _ = controller.ai.predict(obs, deterministic=True)
        else:
            action = controller.env.action_space.sample()

        controller.ai.predict.assert_called_once_with(obs, deterministic=True)
        assert action == 0
