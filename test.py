import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
from typing import MutableMapping
import numpy as np

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
        else:
            action = {1: action}
        observations = env.step(action)
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


if __name__ == "__main__":
    example()