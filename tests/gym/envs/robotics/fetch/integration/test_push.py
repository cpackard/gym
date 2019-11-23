import logging

from gym.envs.robotics.fetch import push


logger = logging.getLogger(__name__)


def test_render_and_reach_goal():
    """
    Basic end-to-end test to show the robot can be rendered, accept actions,
    and report on reaching the goal point
    """
    goal = [1.24, 0.42, 0.425]
    env = push.FetchPushEnv(goal=goal)
    env.render()

    actions = [
        {"joints": ["robot0:shoulder_pan_joint"], "positions": [-2.0]},
        {"joints": ["robot0:shoulder_lift_joint"], "positions": [1.0]},
        {"joints": ["robot0:shoulder_lift_joint"], "positions": [1.0]},
        {"joints": ["robot0:wrist_flex_joint"], "positions": [2.0]},
        {"joints": ["robot0:upperarm_roll_joint"], "positions": [2.0]},
        {"joints": ["robot0:shoulder_pan_joint"], "positions": [-3.0]},
        {"joints": ["robot0:upperarm_roll_joint"], "positions": [4.0]},
        {"joints": ["robot0:upperarm_roll_joint"], "positions": [6.0]},
        {"joints": ["robot0:upperarm_roll_joint"], "positions": [12.0]},
        {"joints": ["robot0:shoulder_pan_joint"], "positions": [-5.0]},
        {"joints": ["robot0:shoulder_pan_joint"], "positions": [-9.0]},
        {"joints": ["robot0:shoulder_pan_joint"], "positions": [-11.0]},
        {"joints": ["robot0:shoulder_lift_joint"], "positions": [2.0]},
        {"joints": ["robot0:shoulder_lift_joint"], "positions": [3.0]},
        {"joints": ["robot0:shoulder_lift_joint"], "positions": [4.0]},
        {"joints": ["robot0:shoulder_lift_joint"], "positions": [5.0]},
        {"joints": ["robot0:shoulder_pan_joint"], "positions": [-7.0]},
        {"joints": ["robot0:shoulder_pan_joint"], "positions": [-5.0]},
    ]

    is_success = False

    for action in actions:
        obs, reward, done, info = env.step(action)
        assert reward == -1.0
        for _ in range(50):
            obs, reward, done, info = env.step({"joints": [], "positions": []})
            if info["is_success"]:
                is_success = True

    env.reset()

    assert is_success
