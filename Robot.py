# !/usr/bin/python3
import sys, getopt
import cv2
import numpy as np
import time
import rospy
from tf.transformations import quaternion_from_euler
from cv_bridge import CvBridge

import habitat
from habitat.config import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, TransformStamped, Transform, Vector3
from sensor_msgs.msg import PointCloud2, PointField, JointState, LaserScan, Image, CameraInfo
from tf import TransformBroadcaster

import myRobotAction
from my_utils import *
from threading import Lock


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


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
        assert sense_freq < action_freq
        self.action_freq = action_freq
        self.action_per_sen = int(action_freq/sense_freq)
        self.count = 0
        self.required_freq = required_freq
        self.lock = Lock()
        
        self.rate = rospy.Rate(action_freq)
        self.bridge = CvBridge()
        self.depth_pubs = [rospy.Publisher(f"{ns}/depth", Image, queue_size=1) for ns in multi_ns]
        self.image_pubs = [rospy.Publisher(f"{ns}/image", Image, queue_size=1) for ns in multi_ns]
        self.camera_info_pubs = [rospy.Publisher(f"{ns}/camera_info", CameraInfo,  queue_size=1) for ns in multi_ns]
        self.gt_tf_pub = TransformBroadcaster()

        # rospy.Timer(1/action_freq, self.action_executor)

    def kick(self, agent_id):
        with self.lock:
            self.t_last_cmd_vel[agent_id] = time.time()

    def action_executor(self):
        while not rospy.is_shutdown():
            self.rate.sleep()
        # with self.lock:
            actions = {}
            current_time = time.time()
            for i in self.agent_ids:
                if current_time - self.t_last_cmd_vel[i] > self.required_freq:
                    continue
                actions[i] = self.action_id
            if actions:
                self.step(actions)
            if self.count % self.action_per_sen == 0:
                _current_time = rospy.Time.from_sec(current_time)
                # publish the collected observations
                obs = [self.get_observation_of(i) for i in range(len(self.agent_ids))]
                for i in self.agent_ids:
                    self.publish_obs(i, obs[i], _current_time)
            self.count += 1

    def publish_obs(self, agent_id, obs: dict, current_time) -> None:
        ns = self.multi_ns[agent_id]
        depth = np.array(obs['depth'] / 10 * 255).astype(np.uint16)
        image = np.array(obs['rgb'])
        _depth = self.bridge.cv2_to_imgmsg(depth)
        _depth.header.stamp = current_time
        _depth.header.frame_id = f"{ns}"
        self.depth_pubs[agent_id].publish(_depth)

        _image = self.bridge.cv2_to_imgmsg(image)
        _image.header = _depth.header
        self.image_pubs[agent_id].publish(_image)

        camera_info = CameraInfo()
        camera_info.header = _depth.header
        camera_info.height = _depth.height
        camera_info.width = _depth.width
        camera_info.distortion_model = "plumb_bob"
        camera_info.D = [0]*5   # All 0, no distortion
        camera_info.K[0] = 570.3422241210938
        camera_info.K[2] = 314.5
        camera_info.K[4] = 570.3422241210938
        camera_info.K[5] = 235.5
        camera_info.K[8] = 1.0
        camera_info.R[0] = 1.0
        camera_info.R[4] = 1.0
        camera_info.R[8] = 1.0
        camera_info.P[0] = 570.3422241210938
        camera_info.P[2] = 314.5
        camera_info.P[5] = 570.3422241210938
        camera_info.P[6] = 235.5
        camera_info.P[10] = 1.0
        self.camera_info_pubs[agent_id].publish(camera_info)

        # Publish ground truth tf of each robot
        trans, quat, _ = get_agent_position(self, agent_id)
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.header.stamp = current_time
        transform.child_frame_id = ns
        transform.transform = Transform(
            translation=Vector3(*trans), rotation=Quaternion(*quat))
        self.gt_tf_pub.sendTransformMessage(transform)

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
        self.init_quat = quaternion_from_euler(0, 0, w)
        self.action_freq = env.action_freq
        set_initial_position(env, agent_name, self.init_trans, self.init_quat)
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
    opts, args = getopt.getopt(argv,"n:a:s:")
    num_robots = 1
    action_freq = 30
    sample_freq = 10

    for opt, arg in opts:
        if opt in '-n':
            num_robots = int(arg)
        elif opt in '-a':
            action_freq = int(arg)
        elif opt in '-s':
            sample_freq = int(arg)

    rospy.init_node("multi_robot_habitat_sim")
    config = habitat.get_config(config_paths=f"configs/tasks/MASLAM{num_robots}.yaml")
    agent_names = config.SIMULATOR.AGENTS
    agent_ids = list(range(len(agent_names)))
    multi_ns = agent_names
    init_poses = [list(getattr(config.SIMULATOR, agent_name).INIT_POSE) for agent_name in agent_names]
    # Avoid overwrite config error
    for agent_name in agent_names:
        del getattr(config.SIMULATOR, agent_name)["INIT_POSE"]

    menv = MultiRobotEnv(config, agent_names, agent_ids,
                         multi_ns, action_id=0, action_freq=action_freq,
                         sense_freq=sample_freq, required_freq=1)
    menv.reset()

    [Robot(menv, agent_names[i], i, 0, i, multi_ns[i], *(init_poses[i])) for i in agent_ids]
    print("Environment creation successful")

    menv.action_executor()
    rospy.spin()

if __name__ == "__main__":
    main(sys.argv[1:])