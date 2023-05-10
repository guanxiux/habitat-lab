from threading import Lock
import math
import numpy as np
from geometry_msgs.msg import Twist
from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
import quaternion
from habitat_sim import ActionSpec
import habitat
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
)
from habitat.utils.geometry_utils import quaternion_to_list

from . import personalized_robot_action

pi_2 = quaternion_from_euler(0,0, math.pi)

twist_lock = Lock()

def set_twist(env, agent_id, action_id, twist: Twist=None, noise=0.):
    twist_lock.acquire()
    if not isinstance(twist.linear, list):
        linear = twist.linear
        twist.linear = np.array(linear.x, linear.y, linear.z)
    x, y, z = np.array(twist.linear)
    if not isinstance(twist.angular, list):
        angular = twist.angular
        twist.angular = np.array([angular.x, angular.y, angular.z])
        # habitat seems to use degree in rotation 
    pitch, roll, yaw = np.rad2deg(twist.angular) * -1
    env.set_agent_action_spec(agent_id, action_id, personalized_robot_action.MyTwist([x, y, z], [pitch, roll, yaw], noise=noise))
    twist_lock.release()

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
    # left-handed coordinate to right-handed coordinate, not sure about correctness
    assert isinstance(quat, quaternion.quaternion)
    quat = quaternion_to_list(quat)
    quat = np.array([-quat[2], -quat[0], quat[1], quat[3]])
    quat = quaternion_multiply(quat, pi_2)
    return quat

def quat_ros_to_habitat(quat):
    quat = list(quat)
    quat = quaternion_multiply(quat, pi_2)
    quat = [quat[1], -quat[2], -quat[0], quat[3]]
    quat = [quat[3]] + quat[:3]
    return quaternion.as_quat_array(quat)

def euler_ros_to_habitat(euler):
    '''
    axis: sxyz in ROS to rzyz in habitat
    '''
    quat = quaternion_from_euler(*euler)
    return np.array(euler_from_quaternion(quat, 'rzyz'))

def euler_habitat_to_ros(euler):
    quat = quaternion_from_euler(*euler, 'rzyz')
    return np.array(euler_from_quaternion(quat))

def set_initial_position(env: habitat.Env, agent_name, trans, rot):
    if len(rot) == 3:
        # Euler angle
        rot = quaternion_from_euler(*rot)
    getattr(env.config.SIMULATOR, agent_name)['START_POSITION'] = trans_ros_to_habitat(trans)
    getattr(env.config.SIMULATOR, agent_name)['START_ROTATION'] = quat_ros_to_habitat(rot)
    env.update_state()

def get_agent_position(env: habitat.Env, agent_id: int=0):
    state = env.get_state_of(agent_id)
    trans = state.position
    trans = trans_habitat_to_ros(trans)
    quat = quat_habitat_to_ros(state.rotation)
    euler = np.array(euler_from_quaternion(quat))
    return trans, quat, euler

def get_sensor_state(env: habitat.Env, agent_id: int=0):
    state = env.get_state_of(agent_id)
    return state.sensor_states
    # return env.sim.get_agent(agent_id).state.sensor_states

def construct(Constructor, args):
    keys = Constructor.get_fields_and_field_types().keys()
    return Constructor(**dict(zip(keys, args)))

def register_my_actions():
    HabitatSimActions.extend_action_space("move_rotate")
    HabitatSimActions.extend_action_space("noisy_move_rotate")
    return "Registered self-defined action: move_rotate, noisy_move_rotate."

def set_action_space(env, agent_id):
    '''
    Set the action_space to personalized two actions
    '''
    agent = env.sim.agents[agent_id]
    agent.agent_config.action_space = {
            0: ActionSpec("move_rotate", Twist()), 
            1: ActionSpec("noisy_move_rotate", Twist())
        }
