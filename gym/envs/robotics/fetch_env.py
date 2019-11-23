from typing import Any, Dict, List, Union

import numpy as np

from gym.envs.robotics import robot_env

from .types import PositionVector


R_GRIPPER = "robot0:r_gripper_finger_joint"
L_GRIPPER = "robot0:l_gripper_finger_joint"

TABLE_X_BOUNDS = [0.9, 1.25]
TABLE_Y_BOUNDS = [0.0, 0.5]
TABLE_Z_POS = 0.425


def goal_distance(goal_a: "np.ndarray", goal_b: "np.ndarray") -> float:
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        block_gripper,
        has_object,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        initial_qpos,
        reward_type,
        goal=None,
        sim=None,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or
              on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            goal [optional] (array with 3 floats): Fixed goal position for testing
            sim [optional] (Simulator): the simulator that runs the environment
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.goal_position = goal

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4, initial_qpos=initial_qpos, sim=sim
        )

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info) -> float:
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            obj_id = self.joint_info[L_GRIPPER].obj_id
            joint_indices = [self.joint_info[L_GRIPPER].joint_index, self.joint_info[R_GRIPPER].joint_index]
            positions = [0.0, 0.0]
            self.sim.set_joint_positions(obj_id, joint_indices, positions)

    def _set_action(self, joints: List[str], positions: "np.ndarray") -> None:
        if not joints:
            return

        assert len(joints) == len(positions), f"Unequal joint name and position lists: {joints}, {positions}"

        obj_id = self.joint_info[joints[0]].obj_id
        joint_indices = [self.joint_info[j].joint_index for j in joints]
        positions *= 0.05  # limit maximum change in position
        self.sim.set_joint_positions(obj_id, joint_indices, positions)

    def _get_obs(self) -> Dict[str, Any]:
        # positions
        grip_pos = np.array(self._get_gripper_xpos())
        dt = self.sim.get_num_substeps() * self.sim.get_timestep()

        joint_info = self.joint_info[L_GRIPPER]
        link_index = joint_info.joint_index  # NOTE in PyBullet links and joints have a 1:1 correspondance
        grip_xvelp, _ = self.sim.get_link_velocity(joint_info.obj_id, link_index)
        grip_velp = np.array(grip_xvelp) * dt

        robot_qpos, robot_qvel = np.zeros(0), np.zeros(0)

        if self.joint_info:
            robot_joints = [j for j in self.joint_info.values() if j.joint_name.startswith("robot")]
            obj_id = robot_joints[0].obj_id  # assumes this is only for a single robot
            joint_indices = [j.joint_index for j in robot_joints]
            robot_qpos, robot_qvel = self.sim.robot_get_obs(obj_id, joint_indices)

        if self.has_object:
            # rotations
            obj_id = self.body_info["object0"]
            pos, rot = self.sim.get_body_position(obj_id, as_euler=True)
            object_pos, object_rot = np.array(pos), np.array(rot)
            # velocities
            base_object_velp, base_object_velr = self.sim.get_body_velocity(obj_id)
            object_velp = np.array(base_object_velp) * dt
            object_velr = np.array(base_object_velr) * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _viewer_setup(self):
        self.sim.reset_debug_cam()

    def _render_callback(self):
        # Visualize target.
        pass

    def _reset_sim(self):
        self.sim.reset_state(self.model_path)

        # Randomize start position of object.
        if self.has_object:
            obj_id = self.body_info["object0"]
            quat = (0.0, 0.0, 0.0, 1.0)
            position = (np.random.uniform(*TABLE_X_BOUNDS), np.random.uniform(*TABLE_Y_BOUNDS), TABLE_Z_POS)

            self.sim.reset_object_position(obj_id, position, quat)

        return True

    def _sample_goal(self) -> "np.ndarray":
        if self.goal_position:
            return np.array(self.goal_position)

        if self.has_object:
            goal = self.initial_gripper_xpos + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal: "np.ndarray", desired_goal: "np.ndarray") -> "np.float32":
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _get_gripper_xpos(self) -> PositionVector:
        """
        NOTE this method is a workaround for the fact that PyBullet doesn't seem to process <site> MuJoCo tags,
        so there's no way to query for the location of the site object at the center of the gripper.
        Instead, we take the sum of the two gripper fingers' position vectors and hope for the best.
        """
        l_grip_pos = self.joint_info[L_GRIPPER].parent_frame_pos
        r_grip_pos = self.joint_info[R_GRIPPER].parent_frame_pos

        assert (
            l_grip_pos[i] == r_grip_pos[i] for i in (0, 2)
        ), f"Grippers incorrectly aligned! left: {l_grip_pos}, right: {r_grip_pos}"

        y_pos_sum = l_grip_pos[1] + r_grip_pos[1]
        return (l_grip_pos[0], y_pos_sum, l_grip_pos[2])

    def _env_setup(self, initial_qpos: Dict[str, Union[float, List[float]]]) -> None:
        for name, value in initial_qpos.items():
            if isinstance(value, list):
                position_vector = tuple(value[:3])
                quaternion = tuple(value[3:])

                obj_id = self.body_info[name]
                self.sim.reset_object_position(obj_id, position_vector, quaternion)  # type: ignore
            else:
                joint = self.joint_info[name]
                self.sim.reset_joint_position(joint.obj_id, joint.joint_index, value)

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + np.array(
            self._get_gripper_xpos()
        )
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        mocap_obj_id = self.body_info.get("robot0:mocap", 0)
        self.sim.reset_object_position(mocap_obj_id, gripper_target, gripper_rotation)
        for _ in range(10):
            self.sim.step_simulation()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = np.array(self._get_gripper_xpos())
        if self.has_object:
            obj_id = self.body_info.get("object0", 0)
            position_vector, _ = self.sim.get_body_position(obj_id)
            self.height_offset = position_vector[2]

    def render(self, mode="human", width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
