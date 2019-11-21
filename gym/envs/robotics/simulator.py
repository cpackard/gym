from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from gym import error

from .types import PositionVector, VelocityVector, Quaternion, ConnectionMode, ControlMode, JointInfo

try:
    import pybullet as pb
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to run pip install -e '.[pybullet]'")


class Simulator:
    """Controller class to communicate with the physics simulator"""

    # Setup / teardown
    # ----------------

    def __init__(self, connection_mode: ConnectionMode = ConnectionMode.GUI):
        """Initialize simulation environment"""
        self.client_id = pb.connect(connection_mode)
        self.n_substeps: Optional[int] = None
        self.timestep = 1.0 / 240.0

    def disconnect(self):
        if self.client_id >= 0:
            pb.disconnect(physicsClientId=self.client_id)

    # Environment setup and details
    # -----------------------------

    def load_environment(self, model_path: str) -> List[int]:
        """Loads the given environment. Return a list of all unique body/object IDs for querying."""
        pb.setGravity(0, 0, -10, physicsClientId=self.client_id)
        object_ids = pb.loadMJCF(model_path, physicsClientId=self.client_id)
        return object_ids

    def save_state(self) -> int:
        state_id = pb.saveState(physicsClientId=self.client_id)
        return state_id

    def restore_state(self, state_id: int) -> None:
        pb.restoreState(stateId=state_id, physicsClientId=self.client_id)

    def get_num_substeps(self) -> int:
        if self.n_substeps is None:
            self.n_substeps = pb.getPhysicsEngineParameters(physicsClientId=self.client_id)["numSubSteps"]

        return self.n_substeps

    def set_num_substeps(self, n_substeps: int) -> None:
        self.n_substeps = n_substeps
        pb.setPhysicsEngineParameter(numSubSteps=n_substeps, physicsClientId=self.client_id)

    def get_timestep(self) -> float:
        return pb.getPhysicsEngineParameters(physicsClientId=self.client_id)["fixedTimeStep"]

    def step_simulation(self) -> None:
        pb.stepSimulation(physicsClientId=self.client_id)

    # GET queries for environment
    # ---------------------------

    def get_initial_joint_info(self, object_ids: List[int]) -> Dict[str, JointInfo]:
        """Create a map of joint name -> joint info for all objects in the environment"""
        joint_info = {}

        for obj_id in object_ids:
            num_joints = pb.getNumJoints(obj_id, physicsClientId=self.client_id)
            obj_joints = (
                JointInfo(*pb.getJointInfo(obj_id, joint_index, physicsClientId=self.client_id), obj_id)  # type: ignore
                for joint_index in range(num_joints)
            )
            joint_info.update({joint.joint_name: joint for joint in obj_joints})

        return joint_info

    def get_initial_body_info(self, object_ids: List[int]) -> Dict[str, int]:
        """Create a map of object name -> object ID"""
        return {
            str(pb.getBodyInfo(obj_id, physicsClientId=self.client_id)[0], "utf-8"): obj_id for obj_id in object_ids
        }

    def robot_get_obs(self, obj_id: int, joint_indices: List[int]) -> Tuple["np.ndarray", "np.ndarray"]:
        """Returns all joint positions and velocities associated with a robot."""
        joint_positions = []
        joint_velocities = []
        for joint_state in pb.getJointStates(obj_id, joint_indices, physicsClientId=self.client_id):
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])

        return (np.array(joint_positions), np.array(joint_velocities))

    def get_latest_joint_info(self, obj_id: int, joint_index: int) -> JointInfo:
        """Gets the latest information for the given joint from the environment"""
        latest_joint_info = JointInfo(  # type: ignore
            *pb.getJointInfo(obj_id, joint_index, physicsClientId=self.client_id), obj_id,
        )
        return latest_joint_info

    def get_body_position(
        self, obj_id: int, as_euler: bool = False
    ) -> Tuple[PositionVector, Union[PositionVector, Quaternion]]:
        position, orientation = pb.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
        if as_euler:
            orientation = pb.getEulerFromQuaternion(orientation, physicsClientId=self.client_id)

        return position, orientation

    def get_body_velocity(self, obj_id: int) -> Tuple[VelocityVector, VelocityVector]:
        linear, angular = pb.getBaseVelocity(obj_id, physicsClientId=self.client_id)
        return np.array(linear), np.array(angular)

    # SET queries for environment
    # ---------------------------

    def reset_object_position(self, obj_id: int, position_vector: PositionVector, quaternion: Quaternion) -> None:
        pb.resetBasePositionAndOrientation(
            obj_id, list(position_vector), list(quaternion), physicsClientId=self.client_id
        )

    def set_joint_positions(
        self,
        obj_id: int,
        joint_indices: List[int],
        positions: "np.ndarray",
        control_mode: int = ControlMode.POSITION_CONTROL.value,
    ) -> None:
        assert len(joint_indices) > 0 and len(joint_indices) == len(
            positions
        ), f"Invalid inputs: {joint_indices}, {positions}"

        pb.setJointMotorControlArray(
            bodyUniqueId=obj_id,
            jointIndices=joint_indices,
            controlMode=control_mode,
            targetPositions=positions,
            physicsClientId=self.client_id,
        )

    def reset_joint_position(self, obj_id: int, joint_index: int, value: float) -> None:
        pb.resetJointState(obj_id, joint_index, value, physicsClientId=self.client_id)

    def get_link_velocity(self, obj_id: int, link_index: int) -> Tuple[VelocityVector, VelocityVector]:
        """Returns the given link's linear and angular Cartesion world velocities"""
        link_state = pb.getLinkState(obj_id, link_index, computeLinkVelocity=1, physicsClientId=self.client_id)
        linear_velocity, angular_velocity = link_state[-2:]
        return linear_velocity, angular_velocity

    # Synthetic Camera Rendering

    def get_camera_image(
        self,
        width: int = 224,
        height: int = 224,
        projection_matrix: Tuple[float] = None,
        view_matrix: Tuple[float] = None,
    ) -> Tuple:
        if projection_matrix is None:
            projection_matrix = pb.computeProjectionMatrixFOV(
                fov=45.0, aspect=1.0, nearVal=0.1, farVal=6, physicsClientId=self.client_id
            )

        if view_matrix is None:
            view_matrix = pb.computeViewMatrix(
                cameraEyePosition=[2, 0.2, 1],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 0, 20],
                physicsClientId=self.client_id,
            )
        return pb.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=projection_matrix)

    def get_debug_cam(self) -> Tuple:
        return pb.getDebugVisualizerCamera(physicsClientId=self.client_id)

    def reset_debug_cam(self) -> None:
        pb.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=60.0,
            cameraPitch=-35.0,
            cameraTargetPosition=[-0.82, 1.5, -0.8],
            physicsClientId=self.client_id,
        )
