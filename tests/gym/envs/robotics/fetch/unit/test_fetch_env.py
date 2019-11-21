import os

import pytest
from unittest.mock import Mock, MagicMock, patch
from hypothesis import given
from hypothesis.strategies import data, integers, floats, lists, text

import numpy as np

from gym.envs.robotics.fetch_env import FetchEnv


class TestFetchEnv:
    """unit tests for fetch_env"""

    @given(test_data=data())
    def test_create_env(self, fetch_env, test_data):
        env = fetch_env(test_data)
        assert env

    @given(test_data=data(), obs=lists(floats()))
    def test_reset_env(self, fetch_env, test_data, obs):
        env = fetch_env(test_data)
        assert env.goal.shape == (3,)

        with patch.object(env, "_get_obs", return_value=obs):
            env.reset()
            assert env.goal.shape == (3,)

    @given(test_data=data(), achieved_goal=lists(floats(), min_size=3, max_size=3))
    def test_step_env(self, fetch_env, test_data, achieved_goal):
        env = fetch_env(test_data)
        joint = test_data.draw(text())
        position = test_data.draw(floats())

        with patch.object(env, "_get_obs", return_value={"achieved_goal": np.array(achieved_goal)}):
            env.joint_info = MagicMock()
            obs, reward, done, info = env.step({"joints": [joint], "positions": [position]})

        assert reward == 0.0 or reward == -1.0
        assert not done
        assert info["is_success"] == 0.0 or info["is_success"] == 1.0

    @pytest.fixture
    def fetch_env(self):
        """Common config for creating a FetchEnv for testing"""

        def _create_fetch_env(test_data):
            sim = Mock(timestep=test_data.draw(integers(min_value=1)), n_substeps=test_data.draw(integers(min_value=1)))
            kwargs = {
                "has_object": False,
                "block_gripper": True,
                "n_substeps": test_data.draw(integers(min_value=1)),
                "gripper_extra_height": test_data.draw(floats(min_value=0.0)),
                "target_in_the_air": False,
                "target_offset": test_data.draw(floats()),
                "obj_range": test_data.draw(floats()),
                "target_range": test_data.draw(floats()),
                "distance_threshold": test_data.draw(floats()),
                "initial_qpos": {},
                "reward_type": "sparse",
                "goal": None,
                "sim": sim,
            }
            model_path = "/some_file.xml"
            obs = {
                "observation": np.array(test_data.draw(lists(floats()))),
                "achieved_goal": np.array(test_data.draw(lists(floats(), min_size=3, max_size=3))),
                "desired_goal": np.array(test_data.draw(lists(floats(), min_size=3, max_size=3))),
            }

            with patch.multiple(
                FetchEnv,
                _get_obs=Mock(return_value=obs),
                _get_gripper_xpos=Mock(return_value=np.array(test_data.draw(lists(floats(), min_size=3, max_size=3)))),
            ):
                fetch_env_obj = FetchEnv(model_path, **kwargs)

            return fetch_env_obj

        return _create_fetch_env

    @pytest.fixture(autouse=True)
    def patch_os_file_check(self):
        """Mock the model-path check in RobotEnv's init method"""
        with patch.object(os.path, "exists", return_value=True) as os_patch:
            yield os_patch
