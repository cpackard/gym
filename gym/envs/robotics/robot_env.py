import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from .simulator import Simulator

DEFAULT_SIZE = 500


class RobotEnv(gym.GoalEnv):
    def __init__(
        self,
        model_path: str,
        initial_qpos: Dict[str, Union[int, List[int]]],
        n_actions: int,
        n_substeps: int,
        sim: Optional[Simulator] = None,
    ):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File {} does not exist".format(fullpath))

        self.model_path = fullpath

        if sim is None:
            sim = Simulator()
        self.sim = sim

        object_ids = self.sim.load_environment(fullpath)
        self.sim.set_num_substeps(n_substeps)

        self.viewer = None
        self._viewers: Dict[str, Any] = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.seed()
        self.joint_info = self.sim.get_initial_joint_info(object_ids)
        self.body_info = self.sim.get_initial_body_info(object_ids)
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state_id = self.sim.save_state()

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space: spaces.Box = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")  # type: ignore
        self.observation_space = spaces.Dict(  # type: ignore
            dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"),
            )
        )

    def __del__(self):
        self.sim.disconnect()

    @property
    def dt(self):
        return self.sim.timestep * self.sim.n_substeps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: Any) -> Tuple[Dict, float, bool, Dict[str, "np.float32"]]:
        joints, positions = action["joints"], np.array(action["positions"])
        self._set_action(joints, positions)
        self.sim.step_simulation()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE) -> Tuple:
        self._render_callback()
        if mode == "rgb_array":
            return self.sim.get_camera_image(width=width, height=height)
        elif mode == "human":
            self.sim.reset_debug_cam()
            return self.sim.get_debug_cam()

        return tuple()

    def _get_viewer(self, mode):
        pass

    # Extension methods
    # ----------------------------

    def _reset_sim(self) -> bool:
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.reset_state(self.model_path)
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, joints: List[str], positions: "np.ndarray") -> None:
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
