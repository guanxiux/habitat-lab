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
import quaternion
from habitat.utils.geometry_utils import quaternion_to_list
from tf.transformations import quaternion_from_euler, euler_from_quaternion

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
        self.linear = np.array([0., 0., 0.]) if trans is None else np.array(trans, dtype=np.float64)
        self.angular = np.array([0., 0., 0.]) if rot is None else np.array(rot, dtype=np.float64)
        self.noise_amount: float = noise

@habitat_sim.registry.register_move_fn(name="move_rotate", body_action=True)
class MoveAndRotate(habitat_sim.SceneNodeControl):
    def __call__(self, scene_node: habitat_sim.SceneNode, twist: Twist) -> None:
        x = twist.linear[0]
        y = twist.linear[1]
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

def set_twist(env, agent_id, action_id, twist: ROS_Twist=None, noise=0.):
    x, y, z = np.array(twist.linear)*-1.
    # habitat seems to use degree in rotation
    pitch, roll, yaw = np.rad2deg(twist.angular)
    env.set_agent_action_spec(agent_id, action_id, Twist(trans=[x, y, z], rot=[pitch, roll, yaw], noise=noise))

def set_my_action_space(env):
    '''
    Set the action_space to personalized two actions
    '''
    HabitatSimActions.extend_action_space("move_rotate")
    HabitatSimActions.extend_action_space("noisy_move_rotate")

    for agent in env.sim.agents:
        agent.agent_config.action_space = {
            0: ActionSpec("move_rotate", Twist()), 
            1: ActionSpec("noisy_move_rotate", Twist())
        }

def trans_habitat_to_ros(trans):
    '''
    The forward direction in habitat is z
    '''
    return np.array([trans[2], trans[0], trans[1]], dtype=np.float64)

def trans_ros_to_habitat(trans):
    return np.array([trans[1], trans[2], trans[0]], dtype=np.float64)

def quat_habitat_to_ros(quat):
    ''' The fucking quaternion package has a quaternion order of w, x, y, z,
        while others' are x,y,z,w
    '''
    assert isinstance(quat, quaternion.quaternion)
    return quaternion_to_list(quat)

def quat_ros_to_habitat(quat):
    quat = list(quat)
    _quat = [quat[3]] + quat[:3]
    return quaternion.as_quat_array(_quat)

def euler_ros_to_habitat(euler):
    '''
    axis: sxyz in ROS to rzyz in habitat
    '''
    quat = quaternion_from_euler(*euler)
    return euler_from_quaternion(quat, 'rzyz')

def euler_habitat_to_ros(euler):
    quat = quaternion_from_euler(*euler, 'rzyz')
    return euler_from_quaternion(quat)

def set_initial_position(env: habitat.Env, agent_name, trans, rot):
    if len(rot) == 3:
        # Euler angle
        rot = euler_ros_to_habitat(rot)
        rot = quaternion_from_euler(*rot)
    getattr(env.config.SIMULATOR, agent_name)['START_POSITION'] = trans_ros_to_habitat(trans)
    getattr(env.config.SIMULATOR, agent_name)['START_ROTATION'] = quat_ros_to_habitat(rot)
    env.update_state()

def get_agent_position(env: habitat.Env, agent_id: int=0):
    state = env.get_state_of(agent_id)
    trans = state.position
    trans = trans_habitat_to_ros(trans)
    quat = quat_habitat_to_ros(state.rotation)
    # left-handed coordinate to right-handed coordinate, not sure about correctness
    quat = [-quat[0], -quat[2], -quat[1], quat[3]]
    euler = np.array(euler_from_quaternion(quat))
    return trans, quat, euler


def test():
    config = habitat.get_config(config_paths="configs/tasks/pointnav.yaml")
    agent_names = config.SIMULATOR.AGENTS
    config.freeze()
    with habitat.Env(config=config) as env:
        env.reset()
        set_my_action_space(env)
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
        for i in range(20):
            env.step({0: 0})
            current_euler += yaw
            trans, _, euler = get_agent_position(env, 0)
            close = np.allclose(quaternion_from_euler(*euler), quaternion_from_euler(0,0,current_euler))
            print(f'Step {i+5}: Agent0 euler {euler[2]} true euler {current_euler} close {close}')
        set_twist(env, 0, 0, Twist([0., 0.2, 0], [0, 0, -np.pi/4]), noise=0.05)
        for i in range(5):
            env.step({0: 0})
            current_euler += -yaw
            trans, _, euler = get_agent_position(env, 0)
            close = np.allclose(quaternion_from_euler(*euler), quaternion_from_euler(0,0,current_euler))
            print(f'Step {i+10}: Agent0 {euler[2]} true euler {current_euler} close {close}')

if __name__ == "__main__":
    test()
