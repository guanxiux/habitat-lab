# !/usr/bin/python3
import cv2
import numpy as np
import time
import rospy
from tf.transformations import quaternion_from_euler
import quaternion

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from geometry_msgs.msg import Twist
FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
agent_1 = [FORWARD_KEY, LEFT_KEY, RIGHT_KEY]
FINISH="f"

FORWARD_KEY_2="i"
LEFT_KEY_2="j"
RIGHT_KEY_2="l"
agent_2 = [FORWARD_KEY_2, LEFT_KEY_2, RIGHT_KEY_2]


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():
    env = habitat.Env(
        config=habitat.get_config("configs/tasks/MASLAM.yaml")
    )

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(FORWARD_KEY_2):
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY_2):
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY_2):
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue
        if chr(keystroke) in agent_1:
            action = {0: action}
            # action = {0: HabitatSimActions.VELOCITY_CONTROL, }
        else:
            action = {1: action}
        observations = env.step(action)
        env.set_agent_action_spec(1, HabitatSimActions.MOVE_FORWARD, 0.01)
        other_obs = env.get_observation_of(1)[1]
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))
        all_obs = np.concatenate((observations["rgb"], other_obs["rgb"]), axis=0)
        cv2.imshow("RGB", transform_rgb_bgr(all_obs))

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.STOP
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


class Robot:
    def __init__(self, env:habitat.Env, agent_name, idx, ns,  x=0, y=0, w=0, action_freq=10, required_freq=1) -> None:
        # x, y, w: initial translation and yaw
        self.idx = idx
        self.ns = ns
        self.init_trans = [x, y, 0.]
        self.init_quat = quaternion_from_euler(0, 0, w)
        self.env = env
        self.action_freq = action_freq
        self.vel_cmd_sub = rospy.Subscriber("cmd_vel",  Twist, self.sub_vel, queue_size=1)
        rospy.Timer(1/action_freq, self.action_executor)
        self.vel = None
        self.move = None
        self.required_freq = required_freq
        self.t_last_cmd_vel = -1

    def sub_vel(self, cmd_vel: Twist):
        self.vel = cmd_vel
        linear_frac = [l / self.action_freq for l in cmd_vel.linear]
        angular_frac = [a /self.action_freq for a in cmd_vel.angular]
        self.move_frac = Twist(linear_frac, angular_frac)
        
        self.t_last_cmd_vel = time.time()
    
    def action_executor(self):
        # Last command received has expired
        if time.time() - self.t_last_cmd_vel > 1 / self.required_freq:
            return
        
        

        

if __name__ == "__main__":
    example()