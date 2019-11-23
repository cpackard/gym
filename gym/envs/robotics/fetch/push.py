import os
from typing import Dict, List, Union
from gym import utils
from gym.envs.robotics import fetch_env

from ..types import PositionVector

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push_full_pybullet.xml")


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type: str = "sparse", goal: PositionVector = None):
        initial_qpos: Dict[str, Union[float, List[float]]] = {}
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            goal=goal,
        )
        utils.EzPickle.__init__(self)
