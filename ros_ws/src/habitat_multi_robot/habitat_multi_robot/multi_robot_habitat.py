#! /usr/bin/env python
import json
from threading import Lock
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from tf_transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist, Pose, Point, Quaternion,\
    TransformStamped, Transform, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo

import habitat
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from .utils import construct, set_initial_position,\
    set_action_space, get_agent_position, set_twist, register_my_actions

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

# DEFAULT_DATASET_JSON = '{"episodes": [{"episode_id": "0", "scene_id": "data/scene_datasets/habitat-test-scenes/apartment_1.glb", "start_position": [-1.2676633596420288, 0.2047852873802185, 12.595427513122559], "start_rotation": [0, 0.4536385088584658, 0, 0.8911857849408661], "info": {"geodesic_distance": 6.335183143615723, "difficulty": "easy"}, "goals": [{"position": [2.2896811962127686, 0.11950381100177765, 16.97636604309082], "radius": null}], "shortest_paths": null, "start_room": null}]}'

DEFAULT_DATASET_JSON = '{"episodes": [{"episode_id": "0", "scene_id": "/habitat-lab/data/Replica/apartment_0/mesh.ply"}]}'

class MultiRobotEnv(habitat.Env, Node):
    '''
    Env to support multi agent action controlled by ROS
    '''
    def __init__(self, testing=False) -> None:
        Node.__init__(self, node_name='multi_robot_habitat',
                      allow_undeclared_parameters=True,
                      automatically_declare_parameters_from_overrides=True)

        num_robots = self.get_parameter_or(
            "/number_of_robots", alternative_value=1).get_parameter_value().integer_value
        action_freq = self.get_parameter_or(
            "/action_frequency", alternative_value=30).get_parameter_value().integer_value
        sense_freq = self.get_parameter_or(
            "/sense_frequency", alternative_value=30).get_parameter_value().integer_value
        required_freq = self.get_parameter_or(
            "/required_frequency", alternative_value=30).get_parameter_value().integer_value
        config_path = self.get_parameter_or(
            "/habitat_config_path",
            alternative_value=f"/habitat-lab/configs/tasks/MASLAM{num_robots}_apartment.yaml").get_parameter_value().string_value
        scene_id = self.get_parameter_or("/habitat_scene_id",
            alternative_value="/habitat-lab/data/Replica/apartment_0/mesh.ply").get_parameter_value().string_value
        self.get_logger().info(f"Number of robots: {num_robots}; action frequency: {action_freq}Hz; sample frequency: {sense_freq}Hz; required frequency: {required_freq}Hz; config path: {config_path}; scene_id: {scene_id}")

        config = habitat.get_config(config_paths=config_path)
        dataset_config = json.loads(DEFAULT_DATASET_JSON)
        for episode in dataset_config['episodes']:
            episode['scene_id'] = scene_id
        dataset = PointNavDatasetV1()
        dataset.from_json(json.dumps(dataset_config))
        register_my_actions()
        habitat.Env.__init__(self, config, dataset=dataset)

        agent_names = config.SIMULATOR.AGENTS
        agent_ids = list(range(len(agent_names)))
        pose_offset = np.array(config.POSITION_OFFSET)

        self.action_id = 0
        self.agent_names = agent_names
        self.agent_ids = agent_ids
        self.multi_ns = agent_names
        self.t_last_cmd_vel = [self.get_clock().now() for i in range(agent_ids)]
        self.last_position = [None] * len(agent_ids)
        self.last_euler = [None] * len(agent_ids)
        assert sense_freq <= action_freq
        self.action_freq = action_freq
        self.sense_freq = sense_freq
        self.action_per_sen = int(action_freq/sense_freq)
        self.count = 0
        self.required_freq = required_freq
        self.lock = Lock()
        self.pos_offset = np.array(pose_offset)
        self.bridge = CvBridge()

        self.pubs = {}
        for ns in self.multi_ns:
            self.pubs[ns] = {
                'depth': self.create_publisher(Image, f"{ns}/camera/depth/image_raw", 1),
                'image': self.create_publisher(Image, f"{ns}/camera/rgb/image_raw", 1),
                'camera_info': self.create_publisher(
                    CameraInfo, f"{ns}/camera/camera_info", 1),
                'odom': self.create_publisher(Odometry, f"{ns}/odom", 1)
            }

        self.br = TransformBroadcaster(self, 10)
        if not testing:
            cb_group = ReentrantCallbackGroup()
            self.create_timer(0.01, self.tf_br, callback_group=cb_group)
            self.create_timer(1./action_freq, self.execute_action, callback_group=cb_group)

    def broadcast_tf(self, current_time, base, target, trans=None, rot=None):
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
            translation=construct(Vector3, trans),
            rotation=construct(Quaternion, quat))
        self.br.sendTransform(transform)

    def kick(self, agent_id):
        with self.lock:
            self.t_last_cmd_vel[agent_id] = self.get_clock().now()

    def execute_action(self):
        with self.lock:
            actions = {}
            current_time = self.get_clock().now()
            for i in self.agent_ids:
                if current_time - self.t_last_cmd_vel[i] > 1 / self.required_freq:
                    continue
                actions[i] = self.action_id
            if actions:
                self.step(actions)
        if self.count % self.action_per_sen == 0:
            # publish the collected observations
            # obs = [self.get_observation_of(i) for i in range(len(self.agent_ids))]
            for i in self.agent_ids:
                # self.publish_obs(i, obs[i], current_time)
                self.publish_obs(i, current_time)
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
        camera_info_msg.k = np.float32([fx, 0, cx, 0, fy, cy, 0, 0, 1])
        camera_info_msg.d = np.float32([0, 0, 0, 0, 0])
        camera_info_msg.p = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        return camera_info_msg

    def publish_obs(self, agent_id, current_time) -> None:
        '''
        Get observation from the simulator and publish.
        '''
        obs = self.get_observation_of(agent_id)

        ns = self.multi_ns[agent_id]
        # shape = obs['depth'].shape
        image = np.array(obs['rgb'])
        # Publish ground truth tf of each robot
        trans, quat, euler = get_agent_position(self, agent_id)
        trans += self.pos_offset
        self.broadcast_tf(current_time, 'map', f'{ns}', trans, quat)
        self.broadcast_tf(current_time, 'map', f'{ns}/odom', trans, quat)
        
        _depth= np.squeeze(obs['depth'], axis=2)
        # Fix empty holes 
        zero_mask = _depth == 0.
        _depth[zero_mask] = 10.
        # move_up = np.vstack([zero_mask[1:], [zero_mask[0]]])
        # move_down = np.vstack([[zero_mask[-1]], zero_mask[:-1]])
        _depth = self.bridge.cv2_to_imgmsg(
            _depth.astype(np.float32), encoding="passthrough"
        )
        _depth.header.stamp = current_time
        _depth.header.frame_id = f"{ns}/camera"
        self.pubs[ns]['depth'].publish(_depth)
        # check = self.bridge.imgmsg_to_cv2(_depth)

        _image = self.bridge.cv2_to_imgmsg(image)
        _image.header = _depth.header
        self.pubs[ns]['image'].publish(_image)

        camera_info = self.make_depth_camera_info_msg(_depth.header, _depth.height, _depth.width)
        self.pubs[ns]['camera_info'].publish(camera_info)

        if self.last_position[agent_id] is None:
            self.last_position[agent_id] = trans
            self.last_euler[agent_id] = euler

        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "map"
        odom.child_frame_id = f"{ns}/odom"
        odom.pose.pose = Pose(
            position=construct(Point, trans), orientation=construct(Quaternion, quat))

        # set the velocity; convert to speed of 1 second
        v_trans = (trans - self.last_position[agent_id])*self.sense_freq
        v_rot = (euler - self.last_euler[agent_id]) * self.sense_freq
        odom.twist.twist = Twist(
            linear=construct(Vector3, v_trans), angular=construct(Vector3, v_rot))
        self.pubs[ns]['odom'].publish(odom)

        self.last_position[agent_id] = trans
        self.last_euler[agent_id] = euler
        return trans, euler

    def tf_br(self, _) -> None:
        with self.lock:
            current_time = self.get_clock().now()
            for agent_id in self.agent_ids:
                ns = self.multi_ns[agent_id]
                trans, quat, _ = get_agent_position(self, agent_id)
                trans += self.pos_offset
                self.broadcast_tf(current_time, "map", f'{ns}/gt_tf', trans, quat)

                # TODO replace tf from map to robot with real SLAM data
                # broadcast_tf(current_time, "map", f'{ns}/odom', trans, quat)
                self.broadcast_tf(current_time, "map", ns, trans, quat)
        

class Robot(Node):
    def __init__(self, env:MultiRobotEnv,idx) -> None:
        # x, y, w: initial translation and yaw
        # habitat config
        agent_name = env.agent_names[idx]
        ns = env.multi_ns[idx]
        Node.__init__(self, node_name=agent_name,
                      allow_undeclared_parameters=True,
                      automatically_declare_parameters_from_overrides=True)
        self.env = env
        self.agent_name = agent_name
        self.agent_id = idx
        self.action_id = 0
        self.action_freq = env.action_freq

        # start_pos = env._sim.sample_navigable_point()
        start_pos = env._sim.pathfinder.get_random_navigable_point()
        start_yaw = np.random.uniform(0, 2 * np.pi)
        set_initial_position(self.env, self.agent_name, start_pos, [0, 0, start_yaw])

        set_action_space(self.env, self.agent_id)
        # ROS config
        self.idx = idx
        self.ns = ns

        self.vel_cmd_sub = self.create_subscription(Twist, f"{ns}/cmd_vel", self.recv_vel, 1)
        self.pose_sub = self.create_subscription(Pose, f"{ns}/pose",  self.recv_pose, 1)

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
        set_initial_position(self.env, self.agent_name, trans, quat)

    def recv_vel(self, cmd_vel: Twist):
        if not isinstance(cmd_vel.linear, list):
            l = cmd_vel.linear
            cmd_vel.linear = [l.x, l.y, l.z]
        if not isinstance(cmd_vel.angular, list):
            a = cmd_vel.angular
            cmd_vel.angular = [a.x, a.y, a.z]

        linear_frac = [l / self.action_freq for l in cmd_vel.linear]
        angular_frac = [a /self.action_freq for a in cmd_vel.angular]
        move_frac = construct(Twist, [construct(Vector3, linear_frac), construct(Vector3, angular_frac)])
        set_twist(self.env, self.agent_id, self.action_id, move_frac)

        self.env.kick(self.agent_id)


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    menv = MultiRobotEnv()
    executor.add_node(menv)
    menv.reset()
    agent_ids = menv.agent_ids

    # Register robot handler to menv
    for i in agent_ids:
        executor.add_node(Robot(menv, idx=i))
    try:
        menv.get_logger().info('Environment creation successful. Shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        menv.get_logger().info('Keyboard interrupt, shutting down.\n')
    for node in executor.get_nodes():
        node.destroy_node()
    # rclpy.spin()
    rclpy.shutdown()


def test(args=None):
    # Testing for step by step debugging
    rclpy.init(args=args)
    menv = MultiRobotEnv()
    menv.reset()
    agent_ids = menv.agent_ids

    # Register robot handler to menv
    robots = []
    for i in agent_ids:
        robots.append(Robot(menv, idx=i))
    v_trans = np.array([0.5, 0., 0.])
    v_rot = np.array([0., 0., 0., 0.])
    twist = Twist(
            linear=construct(Vector3, v_trans),
            angular=construct(Vector3, v_rot))
    robots[0].recv_vel(twist)
    menv.execute_action()
    trans, euler = menv.publish_obs(0, menv.get_clock().now())
    for _ in range(10):
        menv.execute_action()
        _trans, _euler = menv.publish_obs(0, menv.get_clock().now())
        print(f"Speed trans: {np.round((_trans - trans) * menv.action_freq, decimals=2)}, rot: {np.round((_euler - euler)*menv.action_freq, decimals=2)}.")

    for node in robots:
        node.destroy_node()
    menv.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    test()
