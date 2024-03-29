# /usr/bin/env python
import os
import asyncio
import pickle
from functools import partial
import threading
import shlex
import subprocess
import rclpy
import time
from rclpy.node import Node

from rclpy.logging import LoggingSeverity


def get_unused_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr = s.getsockname()
    s.close()
    return int(addr[1])

class RosHabitatInterface(Node):
    def __init__(self, logging_level=LoggingSeverity.INFO,
                 name="ros_interface",
                 addr='127.0.0.1',
                 port=64523) -> None:
        Node.__init__(self, node_name=name)
        self.get_logger().set_level(logging_level)
        self.pubs = {}
        self.subs = {}
        self.subs_msg_store = {}
        self.addr = addr
        self.port = port
        self.get_logger().info("Creating ROS interface.")
        self.server_thread = threading.Thread(target=self.start_server, name="server", daemon=True)
        self.server_thread.start()

    async def handle_req(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                size_msg = await reader.readexactly(4)
                m_size = int.from_bytes(size_msg, byteorder='big')
                data_msg = await reader.readexactly(m_size)
                data: dict = pickle.loads(data_msg)
                req_type = data['req_type']
                addr = writer.get_extra_info('peername')
                self.get_logger().debug(
                    f"{req_type} Received data of {len(data_msg)} bytes from {addr!r}")

                if req_type == 'pub':
                    for pub in data['pubs']:
                        _type, name, msg, qos = pub
                        if name not in self.pubs:
                            self.pubs[name] = self.create_publisher(
                                _type, name, qos)
                        self.pubs[name].publish(msg)
                elif req_type == 'sub':
                    for sub in data['subs']:
                        _type, name, qos = sub
                        if name not in self.subs:
                            func = partial(self.store_sub_msg, name=name)
                            self.subs[name] = self.create_subscription(
                                _type, name, func, qos)
                    write_data = pickle.dumps(self.subs_msg_store)
                    self.subs_msg_store = {}
                    size_msg = len(write_data).to_bytes(4, byteorder='big')
                    writer.write(size_msg + write_data)
                    self.get_logger().debug(
                        f"{req_type} Wrote data of {len(write_data)} bytes to {addr!r}")

                    await writer.drain()
        except KeyboardInterrupt:
            addr = writer.get_extra_info('peername')
            self.get_logger().debug(
                f"Closing connection to {addr!r}.")
            writer.close()
            await writer.wait_closed()

    def store_sub_msg(self, msg, name):
        self.subs_msg_store[name] = msg


    def start_server(self):
        async def _start_server():
            server = await asyncio.start_server(
                self.handle_req, self.addr, self.port)

            addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
            self.get_logger().info(f'Serving on {addrs}')

            async with server:
                await server.serve_forever()
        asyncio.run(_start_server())


def main(args=None):
    rclpy.init(args=args)
    addr = '127.0.0.1'
    port = get_unused_port()
    ros_interface = RosHabitatInterface(addr=addr, port=port)

    ros_interface.declare_parameter("/number_of_robots", 3)
    ros_interface.declare_parameter("/action_frequency", 10.)
    ros_interface.declare_parameter("/sense_frequency", 10.)
    ros_interface.declare_parameter("/required_frequency", 5.)
    ros_interface.declare_parameter("/habitat_config_path",
        "/habitat-lab/configs/ours/MASLAM_apartment_three_robots.yaml")
    ros_interface.declare_parameter("/habitat_scene_id",
        "/habitat-lab/data/Replica/apartment_1.glb")

    num_robots = ros_interface.get_parameter(
        "/number_of_robots").get_parameter_value().integer_value
    action_freq = ros_interface.get_parameter(
        "/action_frequency").get_parameter_value().double_value
    sense_freq = ros_interface.get_parameter(
        "/sense_frequency").get_parameter_value().double_value
    required_freq = ros_interface.get_parameter(
        "/required_frequency").get_parameter_value().double_value
    config_path = ros_interface.get_parameter(
        "/habitat_config_path").get_parameter_value().string_value
    scene_id = ros_interface.get_parameter(
        "/habitat_scene_id").get_parameter_value().string_value

    assert num_robots <= 3, "We currently only support up to three robots."
    assert required_freq < action_freq
    assert sense_freq <= action_freq
    ros_interface.get_logger().info(
        f"""Number of robots: {num_robots};
        action frequency: {action_freq}Hz;
        sample frequency: {sense_freq}Hz;
        required frequency: {required_freq}Hz;
        config path: {config_path};
        scene_id: {scene_id}""")

    dir_path = os.path.dirname(__file__)
    cmd = f"""
    . /opt/conda/etc/profile.d/conda.sh
    conda activate habitat 
    python {dir_path}/multi_robot_habitat.py --number_of_robots {num_robots} --action_frequency {action_freq} --sense_frequency {sense_freq} --required_frequency {required_freq} --habitat_config_path {config_path} --habitat_scene_id {scene_id} --address {addr} --port {port}"""
    habitat = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)
    ros_interface.get_logger().info(f"Executing {cmd}")
    try:
        rclpy.spin(ros_interface)
    except KeyboardInterrupt:
        ros_interface.get_logger().info("Shutting down everything")
        habitat.kill()
        ros_interface.get_logger().info(habitat.communicate()[0])
        ros_interface.destroy_node()

def test(args=None):
    rclpy.init(args=args)
    ros_interface = RosHabitatInterface(logging_level=LoggingSeverity.DEBUG)
    
    try:
        rclpy.spin(ros_interface)
    except KeyboardInterrupt:
        ros_interface.get_logger().info("Shutting down everything")
        ros_interface.destroy_node()

if __name__ == "__main__":
    test()
