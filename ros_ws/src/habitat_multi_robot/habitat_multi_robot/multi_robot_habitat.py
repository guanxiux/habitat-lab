#! /usr/bin/python
import json
import time
from threading import Lock
import asyncio
import pickle
import argparse
import numpy as np
from rclpy.time import Time

import habitat
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from utils.utils import register_my_actions
from Robot import Robot


DEFAULT_DATASET_JSON = '{"episodes": [{"episode_id": "0", "scene_id": "/habitat-lab/data/Replica/apartment_0/mesh.ply", "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1], "info": {"geodesic_distance": 6.335183143615723, "difficulty": "easy"}, "goals": [{"position": [2.2896811962127686, 0.11950381100177765, 16.97636604309082], "radius": null}], "shortest_paths": null, "start_room": null}]}'


class MultiRobotEnv:
    '''
    Env to support multi agent action controlled by ROS
    '''
    def __init__(self, args) -> None:
        self.args = args
        num_robots = args.number_of_robots
        action_freq = args.action_frequency
        sense_freq = args.sense_frequency
        required_freq = args.required_frequency
        config_path = args.habitat_config_path
        scene_id = args.habitat_scene_id

        print(f"Number of robots: {num_robots}; action frequency: {action_freq}Hz; sample frequency: {sense_freq}Hz; required frequency: {required_freq}Hz; config path: {config_path}; scene_id: {scene_id}")

        config = habitat.get_config(config_paths=config_path)
        config.defrost()
        config.SIMULATOR.SCENE = scene_id
        # TODO scene configuration
        config.freeze()
        dataset_config = json.loads(DEFAULT_DATASET_JSON)
        for episode in dataset_config['episodes']:
            episode['scene_id'] = scene_id
        dataset = PointNavDatasetV1()
        dataset.from_json(json.dumps(dataset_config))
        register_my_actions()
        self.sim: HabitatSim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)

        agent_names = config.SIMULATOR.AGENTS[:num_robots]
        agent_ids = list(range(len(agent_names)))[:num_robots]
        pose_offset = np.array(config.POSITION_OFFSET)

        self.action_id = 0
        self.agent_names = agent_names
        self.agent_ids = agent_ids
        self.multi_ns = agent_names
        self.t_last_cmd_vel = [Time(seconds=time.time())] * len(agent_ids)
        self.action_freq = action_freq
        self.sense_freq = sense_freq
        self.action_per_sen = int(action_freq/sense_freq)
        self.count = 0
        self.required_freq = required_freq
        self.lock = Lock()
        self.pos_offset = np.array(pose_offset)

        self.robots = [Robot(self, idx) for idx in agent_ids]
        self.all_subs = {}
        for robot in self.robots:
            self.all_subs.update(robot.subs_cb)

    def generate_pub_msg(self):
        msgs = []
        ts = time.time()
        for robot in self.robots:
            _msgs = robot.publish_obs(Time(seconds=ts), self.sense_freq)
            msgs.extend(_msgs)
        return msgs

    def generate_sub_msg(self):
        msgs = []
        for robot in self.robots:
            msgs.extend(robot.subs_info)
        return msgs

    async def publish(self, _, writer, msg_func, freq=10):
        period = 1. / freq
        while True:
            stime = time.time()

            pub_msgs = {'req_type': 'pub', 'pubs': []}
            pub_msgs["pubs"].extend(msg_func())
            encoded = pickle.dumps(pub_msgs)
            size_msg = len(encoded).to_bytes(4, byteorder='big')
            data = size_msg + encoded

            writer.write(data)
            await writer.drain()
            # print(f"Sending {len(data)} bytes.")
            restime = period - (time.time() - stime)
            if restime > 0.:
                await asyncio.sleep(restime)
            elif restime < 0. and period > 0:
                print(f"Publishing observations is missing desired frequency of {self.sense_freq}")

    async def subscribe(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter, msg_func, freq=10):
        period = 1. / freq
        while True:
            stime = time.time()

            sub_msgs = {'req_type': 'sub', 'subs': []}
            sub_msgs["subs"].extend(msg_func())
            encoded = pickle.dumps(sub_msgs)
            size_msg = len(encoded).to_bytes(4, byteorder='big')
            data = size_msg + encoded

            writer.write(data)
            await writer.drain()

            msize = int.from_bytes(await reader.readexactly(4),
                                   byteorder='big')
            sub_msgs = pickle.loads(await reader.readexactly(msize))
            for key, item in sub_msgs.items():
                assert key in self.all_subs, f"Topic name: {key} unregistered in robots"
                # calling callback func
                self.all_subs[key](item)

            restime = period - (time.time() - stime)
            if restime > 0.:
                await asyncio.sleep(restime)
            elif restime < 0. and period > 0:
                print(f"Subscribing commands is missing desired frequency of {self.sense_freq}")

    async def start_client(self, loop_func, msg_func, freq):
        while True:
            try:
                reader, writer = await asyncio.open_connection(
                    self.args.address, self.args.port)
                await loop_func(reader, writer, msg_func, freq)
            except ConnectionResetError:
                print("Connection reset. Trying to reconnect in 5 seconds.")
                await asyncio.sleep(5)
                continue
            except (KeyboardInterrupt, SystemExit):
                print("Closing connection.")
                writer.close()
                await writer.wait_closed()
                asyncio.get_running_loop().stop()
                break
            except Exception as exception:
                raise exception

    async def execute_action(self, freq):
        period = 1. / freq
        required_period = 1. / self.required_freq
        while True:
            stime = time.time()
            actions = {}
            for i, robot in enumerate(self.robots):
                if robot.vel_cmd_is_stale(required_period):
                    continue
                actions[i] = self.action_id
            if actions:
                self.sim.step(actions)
            restime = period - (time.time() - stime)
            if restime > 0:
                await asyncio.sleep(restime)
            elif restime < 0:
                print(f"Executing Action is missing desired frequency {freq}")

    def register_futures(self):
        self.futures = [
            self.start_client(self.publish, self.generate_pub_msg, freq=self.sense_freq),
            self.start_client(self.subscribe, self.generate_sub_msg, freq=self.action_freq),
            self.execute_action(self.action_freq)                   # exec action
        ]

    def launch_all(self):
        loop = asyncio.get_event_loop()
        print("Launching all processes")
        assert len(self.futures) > 0, "No futures defined"
        result = loop.run_until_complete(asyncio.gather(*self.futures))
        print(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_robots", type=int, default=1)
    parser.add_argument("--action_frequency", type=float, default=30)
    parser.add_argument("--sense_frequency", type=float, default=10)
    parser.add_argument("--required_frequency", type=float, default=5,
        help="Required frequency of action control. Action control older that 1/required_frequency second will be ignored")
    parser.add_argument("--habitat_config_path", type=str,
                        default="/habitat-lab/configs/ours/MASLAM_apartment_three_robots.yaml")
    parser.add_argument("--habitat_scene_id", type=str,
                        default="/habitat-lab/data/Replica/apartment_1.glb")
    parser.add_argument("--address", type=str, default='127.0.0.1',
                        help="IP address of ros interface")
    parser.add_argument("--port", type=int, default=64523, help="Port of ros interface")
    args = parser.parse_args()
    menv = MultiRobotEnv(args=args)
    menv.launch_all()

if __name__ == "__main__":
    main()
