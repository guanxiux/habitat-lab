#!/usr/bin/env python3

r"""
"""

import numpy as np
import magnum as mn
import habitat
import habitat_sim
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Twist
from my_utils import *

_X_AXIS = 0
_Y_AXIS = 1
_Z_AXIS = 2

_rotate_local_fns = [
    habitat_sim.SceneNode.rotate_x_local,
    habitat_sim.SceneNode.rotate_y_local,
    habitat_sim.SceneNode.rotate_z_local,
]

def _rotate_local(
    scene_node: habitat_sim.SceneNode, theta: float, axis: int
) -> None:
    _rotate_local_fns[axis](scene_node, mn.Deg(theta))
    scene_node.rotation = scene_node.rotation.normalized()

def _move_along(scene_node: habitat_sim.SceneNode, distance: float, axis: int) -> None:
    ax = scene_node.transformation[axis].xyz
    scene_node.translate_local(ax * distance)

class MyTwist:
    def __init__(self, trans=None, rot=None, noise=0.) -> None:
        self.linear = np.array([0., 0., 0.]) if trans is None else np.array(trans, dtype=np.float64)
        self.angular = np.array([0., 0., 0.]) if rot is None else np.array(rot, dtype=np.float64)
        self.noise_amount: float = noise

@habitat_sim.registry.register_move_fn(name="move_rotate", body_action=True)
class MoveAndRotate(habitat_sim.SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, twist: MyTwist) -> None:
        x, y = twist.linear[:2]
        w = twist.angular[2]
        # forward x in ROS
        _move_along(scene_node, -x, _Z_AXIS)
        # left y in ROS
        _move_along(scene_node, -y, _X_AXIS)
        # turn right: yaw in ROS
        _rotate_local(
            scene_node, -w, _Y_AXIS
        )

@habitat_sim.registry.register_move_fn(name="noisy_move_rotate", body_action=True)
class NoisyMoveAndRotate(habitat_sim.SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, twist: Twist) -> None:
        noise = twist.noise_amount
        x = np.random.uniform(
            (1 - noise) * twist.linear[0], (1 + noise) * twist.linear[0])
        y = np.random.uniform(
            (1 - noise) * twist.linear[1], (1 + noise) * twist.linear[1])
        w = np.random.uniform(
            (1 - noise) * twist.angular[2], (1 + noise) * twist.angular[2])
        # forward x in ROS
        _move_along(scene_node, -x, _Z_AXIS)
        # left y in ROS
        _move_along(scene_node, -y, _X_AXIS)
        # turn right: yaw in ROS
        _rotate_local(
            scene_node, -w, _Y_AXIS
        )


if __name__ == "__main__":    
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    agent_names = config.SIMULATOR.AGENTS
    config.freeze()
    with habitat.Env(config=config) as env:
        env.reset()
        set_my_action_space(env, 0)
        set_initial_position(env, agent_names[0], [4, 0, 0], [0, 0, 0])
        trans, _, euler = get_agent_position(env, 0)
        print(f'Step 0: Agent0 trans {trans} euler {euler}')
        set_twist(env, 0, 0, Twist([0.2, 0, 0], [0, 0, 0]))
        for i in range(5):
            env.step({0: 0})
            trans, _, euler = get_agent_position(env, 0)
            print(f'Step {i}: Agent0 trans {trans} euler {euler}')
        yaw = np.pi/4
        set_twist(env, 0, 0, Twist([0, 0, 0], [0, 0, yaw]))
        current_euler = 0
        for i in range(8):
            env.step({0: 0})
            current_euler += yaw
            trans, _, euler = get_agent_position(env, 0)
            close = np.allclose(quaternion_from_euler(*euler), quaternion_from_euler(0,0,current_euler), atol=1e-4) or np.allclose(-1*quaternion_from_euler(*euler), quaternion_from_euler(0,0,current_euler), atol=1e-4)
            print(f'Step {i+5}: Agent0 euler {euler} true euler {current_euler} close {close}')
        set_twist(env, 0, 0, Twist([0., 0.2, 0], [0, 0, -np.pi/4]), noise=0.05)
        for i in range(5):
            env.step({0: 0})
            current_euler += -yaw
            trans, _, euler = get_agent_position(env, 0)
            close = np.allclose(quaternion_from_euler(*euler), quaternion_from_euler(0,0,current_euler), atol=1e-4) or np.allclose(-1*quaternion_from_euler(*euler), quaternion_from_euler(0,0,current_euler), atol=1e-4)
            print(f'Step {i+13}: Agent0 trans {trans} euler {euler[2]} true euler {current_euler} close {close}')
        env.close()
