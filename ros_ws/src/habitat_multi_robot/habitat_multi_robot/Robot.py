#! /usr/bin/python
import copy
import time
import numpy as np
from rclpy.time import Time
from tf_transformations import quaternion_from_euler
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Pose, Point, Quaternion,\
    TransformStamped, Transform,Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

from utils.utils import construct, set_initial_position,\
    set_action_space, get_agent_position, set_twist


def fake_publish(publisher_info, msg, pub_msg_list: list):
    publish_msg = copy.copy(publisher_info)
    publish_msg[-2] = msg
    pub_msg_list.append(publish_msg)

def fake_br_tf(current_time, base, target, trans=None, rot=None, msg_list=None):
    if trans is None:
        trans = [0., 0., 0.]
    if rot is None:
        quat = quaternion_from_euler(0, 0, 0)
    elif len(rot) == 3:
        quat = quaternion_from_euler(*rot)
    else:
        quat = rot
    transform = TransformStamped()
    transform.header.frame_id = base
    transform.header.stamp = current_time.to_msg()
    transform.child_frame_id = target
    transform.transform = Transform(
        translation=construct(Vector3, trans),
        rotation=construct(Quaternion, quat))
    msg_list.append([TransformStamped, '/tf', transform, 1])

def make_depth_camera_info_msg(header, height, width):
    r"""
    Create camera info message for depth camera.
    :param header: header to create the message
    :param height: height of depth image
    :param width: width of depth image
    :returns: camera info message of type CameraInfo.
    """
    # code modifed upon work by Bruce Cui
    camera_info_msg = CameraInfo()
    camera_info_msg.header = header
    fx, fy = width / 2, height / 2
    cx, cy = width / 2, height / 2

    camera_info_msg.width = width
    camera_info_msg.height = height
    camera_info_msg.distortion_model = "plumb_bob"
    camera_info_msg.k = np.float64([fx, 0, cx, 0, fy, cy, 0, 0, 1])
    camera_info_msg.d = [0., 0., 0., 0., 0.]
    camera_info_msg.p = np.float64([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0])
    return camera_info_msg

class Robot:
    '''
    Robot instance that publishes observation and subscribes to velocity commands.
    '''
    def __init__(self, env, idx) -> None:
        # x, y, w: initial translation and yaw
        # habitat config
        agent_name = env.agent_names[idx]
        ns = env.multi_ns[idx]
        sim: HabitatSim = env.sim
        self.env = env
        self.sim: HabitatSim = sim
        self.agent_name = agent_name
        self.agent_id = idx
        self.action_id = 0
        self.action_freq = env.action_freq
        self.last_position = None
        self.last_euler = None
        self.time_last_vel_cmd = Time(seconds=time.time())

        start_pos = self.sim.sample_navigable_point()
        start_yaw = np.random.uniform(0, 2 * np.pi)
        set_initial_position(self.sim, self.agent_name, start_pos, [0, 0, start_yaw])
        print(f"{agent_name} starting at position {np.round(start_pos, decimals=2)} yaw {start_yaw:.2f}")

        set_action_space(self.sim, self.agent_id)
        # ROS config
        self.idx = idx
        self.ns = ns
        self.bridge = CvBridge()
        
        self.pubs = {
            'depth': [Image, f"{ns}/camera/depth/image_raw", None, 1],
            'image': [Image, f"{ns}/camera/rgb/image_raw", None, 1],
            'camera_info': [CameraInfo, f"{ns}/camera/camera_info", None, 1],
            'odom': [Odometry, f"{ns}/odom", None, 1],
            'tf': [TransformStamped, '/tf', None, 1]
        }
        self.subs_info = [
            [Twist, f"{ns}/cmd_vel", self.recv_vel, 1],
            [Pose, f"{ns}/pose",  self.recv_pose, 1]
        ]
        self.subs_cb = {
            f"{ns}/cmd_vel": self.recv_vel,
            f"{ns}/pose": self.recv_pose
        }
        self.camera_info_msg = None

    def vel_cmd_is_stale(self, required_period):
        return time.time() - self.time_last_vel_cmd > required_period

    def recv_pose(self, pose: Pose):
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        trans = np.array([x,y,z])
        _x = pose.orientation.x
        _y = pose.orientation.y
        _z = pose.orientation.z
        _w = pose.orientation.w
        quat = np.array([_x, _y, _z, _w])
        set_initial_position(self.sim, self.agent_name, trans, quat)

    def recv_vel(self, cmd_vel: Twist):
        linear = cmd_vel.linear
        angular = cmd_vel.angular
        if isinstance(linear, Vector3):
            linear = np.array([linear.x, linear.y, linear.z])
        if isinstance(angular, Vector3):
            angular = np.array([angular.x, angular.y, angular.z])

        linear_frac = linear / self.action_freq
        angular_frac = angular / self.action_freq
        set_twist(self.sim, self.agent_id, self.action_id, linear_frac, angular_frac)
        self.time_last_vel_cmd = time.time()

    def publish_obs(self, current_time: Time, sense_freq=30) -> None:
        '''
        Get observation from the simulator and publish.
        '''
        msgs = []
        obs = self.sim.get_sensor_observations(self.agent_id)
        ns = self.ns
        trans, quat, euler = get_agent_position(self.sim, self.agent_id)
        fake_br_tf(current_time, "map", ns, trans, quat, msgs)
        fake_br_tf(current_time, 'map', f'{ns}/odom', trans, quat, msgs)

        depth= np.array(obs['depth'])
        # Fix empty holes 
        # zero_mask = depth == 0.
        # depth[zero_mask] = 10.
        depth = self.bridge.cv2_to_imgmsg(
            depth.astype(np.float32), encoding="passthrough"
        )
        depth.header.stamp = current_time.to_msg()
        depth.header.frame_id = f"{ns}/camera"
        fake_publish(self.pubs['depth'], depth, msgs)

        image = np.array(obs['rgb'])
        image = self.bridge.cv2_to_imgmsg(image)
        image.header = depth.header
        fake_publish(self.pubs['image'], image, msgs)

        if self.camera_info_msg is None:
            self.camera_info_msg = make_depth_camera_info_msg(
                depth.header, depth.height, depth.width)
        fake_publish(self.pubs['camera_info'], self.camera_info_msg, msgs)

        if self.last_position is None:
            self.last_position = trans
            self.last_euler = euler

        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = f"{ns}/odom"
        odom.pose.pose = Pose(
            position=construct(Point, trans), orientation=construct(Quaternion, quat))

        # set the velocity; convert to speed of 1 second
        v_trans = (trans - self.last_position)* sense_freq
        v_rot = (euler - self.last_euler) * sense_freq
        odom.twist.twist = Twist(
            linear=construct(Vector3, v_trans), angular=construct(Vector3, v_rot))
        fake_publish(self.pubs[ns]['odom'], odom, msgs)

        self.last_position = trans
        self.last_euler = euler
        return msgs