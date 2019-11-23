from enum import IntEnum
from dataclasses import dataclass, field, InitVar

from gym import error

from .generic import PositionVector

try:
    import pybullet as pb
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to run pip install -e '.[pybullet]'")


class ConnectionMode(IntEnum):
    GUI = pb.GUI
    DIRECT = pb.DIRECT
    SHARED_MEMORY = pb.SHARED_MEMORY
    UDP = pb.UDP
    TCP = pb.TCP


class ControlMode(IntEnum):
    VELOCITY_CONTROL = pb.VELOCITY_CONTROL
    TORQUE_CONTROL = pb.TORQUE_CONTROL
    POSITION_CONTROL = pb.POSITION_CONTROL


class JointType(IntEnum):
    JOINT_REVOLUTE = pb.JOINT_REVOLUTE
    JOINT_PRISMATIC = pb.JOINT_PRISMATIC
    JOINT_SPHERICAL = pb.JOINT_SPHERICAL
    JOINT_PLANAR = pb.JOINT_PLANAR
    JOINT_FIXED = pb.JOINT_FIXED


@dataclass
class JointInfo:
    joint_index: int
    _joint_name: InitVar[str]
    _joint_type: InitVar[int]
    q_index: int
    u_index: int
    flags: int
    joint_damping: float
    joint_friction: float
    joint_lower_limit: float
    joint_upper_limit: float
    joint_max_force: float
    joint_max_velocity: float
    link_name: float
    joint_axis: PositionVector
    parent_frame_pos: PositionVector
    parent_frame_orn: PositionVector
    parent_index: int
    obj_id: int

    joint_name: str = field(init=False)
    joint_type: JointType = field(init=False)

    def __post_init__(self, _joint_name, _joint_type):
        self.joint_name = str(_joint_name, "utf-8")
        self.joint_type = JointType(_joint_type)

    def __str__(self):
        return f"{self.joint_index}-{self.joint_name}-{self.joint_type.name}"
