# !/usr/bin/python3
import sys, getopt
import os
import cv2
import numpy as np
import time
import rospy
from tf.transformations import quaternion_from_euler
from cv_bridge import CvBridge

import habitat
from habitat.config import Config
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, TransformStamped, Transform, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, JointState, LaserScan, Image, CameraInfo
from tf import TransformBroadcaster

import myRobotAction
from my_utils import *
from threading import Lock


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


br = TransformBroadcaster()

def broadcast_tf(current_time, base, target, trans=None, rot=None):
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
    transform.header.stamp = current_time
    transform.child_frame_id = target
    transform.transform = Transform(
        translation=Vector3(*trans), rotation=Quaternion(*quat))
    br.sendTransformMessage(transform)

class MultiRobotEnv(habitat.Env):
    '''
    Env to support multi agent action controlled by ROS
    '''
    def __init__(self, config: Config, agent_names, agent_ids, multi_ns,
                 action_id, action_freq=30, sense_freq=10, required_freq=1) -> None:
        super().__init__(config=config)

        self.action_id = action_id
        self.agent_names = agent_names
        self.agent_ids = agent_ids
        self.multi_ns = multi_ns
        self.t_last_cmd_vel = [-1 * required_freq] * len(agent_ids)
        self.last_position = [None] * len(agent_ids)
        self.last_euler = [None] * len(agent_ids)
        assert sense_freq < action_freq
        self.action_freq = action_freq
        self.sense_freq = sense_freq
        self.action_per_sen = int(action_freq/sense_freq)
        self.count = 0
        self.required_freq = required_freq
        self.lock = Lock()
        
        self.rate = rospy.Rate(action_freq)
        self.bridge = CvBridge()
        self.depth_pubs = [rospy.Publisher(f"{ns}/camera/depth/image_raw", Image, queue_size=1) for ns in multi_ns]
        self.image_pubs = [rospy.Publisher(f"{ns}/camera/rgb/image_raw", Image, queue_size=1) for ns in multi_ns]
        self.camera_info_pubs = [rospy.Publisher(f"{ns}/camera/depth/camera_info", CameraInfo,  queue_size=1) for ns in multi_ns]
        self.odom_pub = [rospy.Publisher(f"{ns}/odom", Odometry, queue_size=1) for ns in multi_ns]

        self.timer = rospy.Timer(rospy.Duration(0.05), self.tf_br)
        # rospy.Timer(1/action_freq, self.action_executor)

    def kick(self, agent_id):
        with self.lock:
            self.t_last_cmd_vel[agent_id] = time.time()

    def action_executor(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
            with self.lock:
                actions = {}
                current_time = time.time()
                self.current_time = rospy.Time.from_sec(current_time)
                for i in self.agent_ids:
                    if current_time - self.t_last_cmd_vel[i] > 1 / self.required_freq:
                        continue
                    actions[i] = self.action_id
                if actions:
                    self.step(actions)
            if self.count % self.action_per_sen == 0:
                # publish the collected observations
                obs = [self.get_observation_of(i) for i in range(len(self.agent_ids))]
                for i in self.agent_ids:
                    self.publish_obs(i, obs[i], self.current_time)
            self.count += 1

    def make_depth_camera_info_msg(self, header, height, width):
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
        camera_info_msg.K = np.float32([fx, 0, cx, 0, fy, cy, 0, 0, 1])
        camera_info_msg.D = np.float32([0, 0, 0, 0, 0])
        camera_info_msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        return camera_info_msg

    def publish_obs(self, agent_id, obs: dict, current_time) -> None:
        ns = self.multi_ns[agent_id]
        # shape = obs['depth'].shape
        image = np.array(obs['rgb'])
        # Publish ground truth tf of each robot
        trans, quat, euler = get_agent_position(self, agent_id)
        br.sendTransform(trans, quat, current_time, f'{ns}', 'map')
        br.sendTransform(trans, quat, current_time, f'{ns}/odom', 'map')
        
        _depth= np.squeeze(obs['depth'], axis=2)
        _depth = self.bridge.cv2_to_imgmsg(
            _depth.astype(np.float32), encoding="passthrough"
        )
        _depth.header.stamp = current_time
        _depth.header.frame_id = f"{ns}"
        self.depth_pubs[agent_id].publish(_depth)
        # check = self.bridge.imgmsg_to_cv2(_depth)

        _image = self.bridge.cv2_to_imgmsg(image)
        _image.header = _depth.header
        self.image_pubs[agent_id].publish(_image)

        camera_info = self.make_depth_camera_info_msg(_depth.header, _depth.height, _depth.width)

        self.camera_info_pubs[agent_id].publish(camera_info)

        
        if self.last_position[agent_id] is None:
            self.last_position[agent_id] = trans
            self.last_euler[agent_id] = euler

        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "map"
        odom.child_frame_id = f"{ns}/odom"

        # set the position
        odom.pose.pose = Pose(Point(*trans), Quaternion(*quat))

        # set the velocity
        v_trans = (trans - self.last_position[agent_id])*self.sense_freq
        v_rot = (euler - self.last_euler[agent_id]) * self.sense_freq
        odom.twist.twist = Twist(Vector3(*v_trans), Vector3(*v_rot))

        # publish the message
        self.odom_pub[agent_id].publish(odom)
            
        self.last_position[agent_id] = trans
        self.last_euler[agent_id] = euler

    def tf_br(self, _) -> None:
        with self.lock:
            current_time = rospy.Time.now()
            for agent_id in self.agent_ids:
                ns = self.multi_ns[agent_id]
                trans, quat, _ = get_agent_position(self, agent_id)
                
                broadcast_tf(current_time, "map", f'{ns}/gt_tf', trans, quat)

                # TODO replace tf from map to robot with real SLAM data
                broadcast_tf(current_time, "map", f'{ns}/odom', trans, quat)
                broadcast_tf(current_time, "map", ns, trans, quat)
        

class Robot:
    def __init__(self, env:MultiRobotEnv, agent_name,
                 agent_id, action_id,
                 idx, ns, x=0, y=0, w=0) -> None:
        # x, y, w: initial translation and yaw
        # habitat config
        self.env = env
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.action_id = action_id
        self.init_trans = [x, y, 0.]
        self.init_rot = [0, 0, w]
        self.action_freq = env.action_freq
        set_initial_position(env, agent_name, self.init_trans, self.init_rot)
        set_my_action_space(env, agent_id)

        # ROS config
        self.idx = idx
        self.ns = ns

        self.vel_cmd_sub = rospy.Subscriber(f"{ns}/cmd_vel",  Twist, self.sub_vel, queue_size=1)

    def sub_vel(self, cmd_vel: Twist):
        if not isinstance(cmd_vel.linear, list):
            l = cmd_vel.linear
            cmd_vel.linear = [l.x, l.y, l.z]
        if not isinstance(cmd_vel.angular, list):
            a = cmd_vel.angular
            cmd_vel.angular = [a.x, a.y, a.z]

        linear_frac = [l / self.action_freq for l in cmd_vel.linear]
        angular_frac = [a /self.action_freq for a in cmd_vel.angular]

        move_frac = Twist(linear_frac, angular_frac)
        set_twist(self.env, self.agent_id, self.action_id, move_frac)

        self.env.kick(self.agent_id)

def main(argv):
    opts, args = getopt.getopt(argv,"d:c:n:a:s:r:")
    num_robots = 1
    action_freq = 30
    sample_freq = 10
    required_freq = 5
    config_path = ""
    # habitat_dir = "/home/guan/ros_ws/habitat-lab"
    os.chdir(os.path.dirname(__file__))
    for opt, arg in opts:
        if opt in '-n':
            num_robots = int(arg)
        elif opt in '-a':
            action_freq = int(arg)
        elif opt in '-d':
            habitat_dir = arg
        elif opt in '-s':
            sample_freq = int(arg)
        elif opt in '-r':
            required_freq = int(arg)
        elif opt in '-c':
            config_path = arg
    if not config_path:
        config_path = f"configs/tasks/MASLAM{num_robots}.yaml"
    rospy.init_node("multi_robot_habitat_sim")
    config = habitat.get_config(config_paths=config_path)
    agent_names = config.SIMULATOR.AGENTS
    agent_ids = list(range(len(agent_names)))
    multi_ns = agent_names
    init_poses = [list(getattr(config.SIMULATOR, agent_name).INIT_POSE) for agent_name in agent_names]
    # Avoid overwrite config error
    for agent_name in agent_names:
        del getattr(config.SIMULATOR, agent_name)["INIT_POSE"]

    menv = MultiRobotEnv(config, agent_names, agent_ids,
                         multi_ns, action_id=0, action_freq=action_freq,
                         sense_freq=sample_freq, required_freq=required_freq)
    menv.reset()

    [Robot(menv, agent_names[i], i, 0, i, multi_ns[i], *(init_poses[i])) for i in agent_ids]
    print("Environment creation successful")

    menv.action_executor()
    rospy.spin()

if __name__ == "__main__":
    main(sys.argv[1:])