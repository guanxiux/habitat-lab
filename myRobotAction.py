#!/usr/bin/env python3

r"""
"""

import numpy as np
import magnum as mn
import habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
)
from geometry_msgs.msg import Twist as ROS_Twist

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

# @attr.s(auto_attribs=True, slots=True)
class Twist:
    def __init__(self, trans=None, rot=None, noise=0.) -> None:
        if trans is None:
            self.trans = np.array([0., 0., 0.])
        else:
            self.trans = np.array(trans)
        if rot is None:
            self.rot = np.array([0., 0., 0.])
        else:
            self.rot = np.array(rot)
        self.noise_amount: float = noise

@habitat_sim.registry.register_move_fn(name="move_rotate", body_action=True)
class MoveAndRotate(habitat_sim.SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, twist: Twist) -> None:
        x = twist.trans[0]
        y = twist.trans[1]
        w = twist.rot[2]
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
            (1 - noise) * twist.trans[0], (1 + noise) * twist.trans[0])
        y = np.random.uniform(
            (1 - noise) * twist.trans[1], (1 + noise) * twist.trans[1])
        w = np.random.uniform(
            (1 - noise) * twist.rot[2], (1 + noise) * twist.rot[2])
        # forward x in ROS
        _move_along(scene_node, -x, _Z_AXIS)
        # left y in ROS
        _move_along(scene_node, -y, _X_AXIS)
        # turn right: yaw in ROS
        _rotate_local(
            scene_node, -w, _Y_AXIS
        )

def set_twist(env, agent_id, action_id, twist: ROS_Twist=None, noise=0.):
    if isinstance(twist, ROS_Twist):
        x, y, z = twist.linear
        pitch, roll, yaw = twist.angular
        env.set_agent_action_spec(agent_id, action_id, Twist(trans=[x, y, z], rot=[pitch, roll, yaw], noise=noise))
    elif isinstance(twist, Twist):
        env.set_agent_action_spec(agent_id, action_id, twist)

def set_my_action_space(env):
    HabitatSimActions.extend_action_space("move_rotate")
    HabitatSimActions.extend_action_space("noisy_move_rotate")

    for agent in env.sim.agents:
        agent.agent_config.action_space = {
            0: ActionSpec("move_rotate", Twist()), 
            1: ActionSpec("noisy_move_rotate", Twist())
        }

def get_agent_position(env: habitat.Env, agent_id: int=0):
    env.get_state_of(agent_id)

def test():
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    config.freeze()
    with habitat.Env(config=config) as env:
        env.reset()
        set_my_action_space(env)
        set_twist(env, 0, 0, Twist([0.2, 0, 0], [0, 0, 10]))
        env.step({0: 0})
        
        set_twist(env, 0, 1, ROS_Twist([0.2, 0, 0], [0, 0, 10]), noise=0.05)
        env.step({0: 1})

if __name__ == "__main__":
    test()
