#! /usr/bin/env python
import json
from threading import Lock
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
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
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from utils import construct, set_initial_position,\
    set_action_space, get_agent_position, set_twist, register_my_actions

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

DEFAULT_DATASET_JSON = '{"episodes": [{"episode_id": "0", "scene_id": "/habitat-lab/data/Replica/apartment_0/mesh.ply", "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1], "info": {"geodesic_distance": 6.335183143615723, "difficulty": "easy"}, "goals": [{"position": [2.2896811962127686, 0.11950381100177765, 16.97636604309082], "radius": null}], "shortest_paths": null, "start_room": null}]}'

# DEFAULT_DATASET_JSON = '{"episodes": [{"episode_id": "0", "scene_id": "/habitat-lab/data/Replica/apartment_0/mesh.ply"}]}'

class MultiRobotEnv(Node):
    '''
    Env to support multi agent action controlled by ROS
    '''
    def __init__(self, testing=False) -> None:
        Node.__init__(self, node_name='multi_robot_habitat',
                      allow_undeclared_parameters=True,
                      automatically_declare_parameters_from_overrides=True)
        self.declare_parameter("/number_of_robots", 3)
        self.declare_parameter("/action_frequency", 30)
        self.declare_parameter("/sense_frequency", 30)
        self.declare_parameter("/required_frequency", 30)
        self.declare_parameter("/habitat_config_path",
            "/habitat-lab/configs/ours/MASLAM_apartment_three_robots.yaml")
        self.declare_parameter("/habitat_scene_id",
            "/habitat-lab/data/Replica/apartment_1.glb")

        num_robots = self.get_parameter(
            "/number_of_robots").get_parameter_value().integer_value
        action_freq = self.get_parameter(
            "/action_frequency").get_parameter_value().integer_value
        sense_freq = self.get_parameter(
            "/sense_frequency").get_parameter_value().integer_value
        required_freq = self.get_parameter(
            "/required_frequency").get_parameter_value().integer_value
        config_path = self.get_parameter(
            "/habitat_config_path").get_parameter_value().string_value
        scene_id = self.get_parameter(
            "/habitat_scene_id").get_parameter_value().string_value

        assert num_robots <= 3, "We currently only support up to three robots."
        self.get_logger().info(f"Number of robots: {num_robots}; action frequency: {action_freq}Hz; sample frequency: {sense_freq}Hz; required frequency: {required_freq}Hz; config path: {config_path}; scene_id: {scene_id}")

        config = habitat.get_config(config_paths=config_path)
        config.defrost()
        config.SIMULATOR.SCENE = scene_id
        config.freeze()
        dataset_config = json.loads(DEFAULT_DATASET_JSON)
        for episode in dataset_config['episodes']:
            episode['scene_id'] = scene_id
        dataset = PointNavDatasetV1()
        dataset.from_json(json.dumps(dataset_config))
        register_my_actions()
        self.sim: HabitatSim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)
        # habitat.Env.__init__(self, config, dataset=dataset)

        agent_names = config.SIMULATOR.AGENTS
        agent_ids = list(range(len(agent_names)))
        pose_offset = np.array(config.POSITION_OFFSET)

        self.action_id = 0
        self.agent_names = agent_names
        self.agent_ids = agent_ids
        self.multi_ns = agent_names
        self.t_last_cmd_vel = [self.get_clock().now() for _ in agent_ids]
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
        transform.header.stamp = current_time.to_msg()
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
                if current_time - self.t_last_cmd_vel[i] > \
                    Duration(seconds =1 / self.required_freq):
                    continue
                actions[i] = self.action_id
            if actions:
                self.sim.step(actions)
        if self.count % self.action_per_sen == 0:
            # publish the collected observations
            for i in self.agent_ids:
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
        camera_info_msg.k = np.float64([fx, 0, cx, 0, fy, cy, 0, 0, 1])
        camera_info_msg.d = [0., 0., 0., 0., 0.]
        camera_info_msg.p = np.float64([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0])
        return camera_info_msg

    def publish_obs(self, agent_id, current_time) -> None:
        '''
        Get observation from the simulator and publish.
        '''
        obs = self.sim.get_sensor_observations(agent_id)
        ns = self.multi_ns[agent_id]
        # shape = obs['depth'].shape
        # Publish ground truth tf of each robot
        trans, quat, euler = get_agent_position(self.sim, agent_id)
        trans += self.pos_offset
        self.broadcast_tf(current_time, 'map', f'{ns}', trans, quat)
        self.broadcast_tf(current_time, 'map', f'{ns}/odom', trans, quat)
        
        depth= np.array(obs['depth'])
        # Fix empty holes 
        zero_mask = depth == 0.
        depth[zero_mask] = 10.
        # move_up = np.vstack([zero_mask[1:], [zero_mask[0]]])
        # move_down = np.vstack([[zero_mask[-1]], zero_mask[:-1]])
        depth = self.bridge.cv2_to_imgmsg(
            depth.astype(np.float32), encoding="passthrough"
        )
        depth.header.stamp = current_time.to_msg()
        depth.header.frame_id = f"{ns}/camera"
        self.pubs[ns]['depth'].publish(depth)
        # check = self.bridge.imgmsg_to_cv2(_depth)

        image = np.array(obs['rgb'])
        image = self.bridge.cv2_to_imgmsg(image)
        image.header = depth.header
        self.pubs[ns]['image'].publish(image)

        camera_info = self.make_depth_camera_info_msg(depth.header, depth.height, depth.width)
        self.pubs[ns]['camera_info'].publish(camera_info)

        if self.last_position[agent_id] is None:
            self.last_position[agent_id] = trans
            self.last_euler[agent_id] = euler

        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
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
    def __init__(self, env: MultiRobotEnv, idx) -> None:
        # x, y, w: initial translation and yaw
        # habitat config
        agent_name = env.agent_names[idx]
        ns = env.multi_ns[idx]
        Node.__init__(self, node_name=agent_name,
                      allow_undeclared_parameters=True,
                      automatically_declare_parameters_from_overrides=True)
        sim = env.sim
        self.env = env
        self.sim = sim
        self.agent_name = agent_name
        self.agent_id = idx
        self.action_id = 0
        self.action_freq = env.action_freq

        # start_pos = sim._sim.sample_navigable_point()
        start_pos = sim.pathfinder.get_random_navigable_point()
        start_yaw = np.random.uniform(0, 2 * np.pi)
        set_initial_position(self.sim, self.agent_name, start_pos, [0, 0, start_yaw])

        set_action_space(self.sim, self.agent_id)
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

        self.env.kick(self.agent_id)


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    menv = MultiRobotEnv()
    executor.add_node(menv)
    agent_ids = menv.agent_ids

    # Register robot handler to menv
    for i in agent_ids:
        executor.add_node(Robot(menv.sim, idx=i))
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
    agent_ids = menv.agent_ids

    # Register robot handler to menv
    robots = []
    for i in agent_ids:
        robots.append(Robot(menv, idx=i))
    v_trans = np.array([0.5, 0., 0.])
    v_rot = np.array([0., 0., 0.])
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
