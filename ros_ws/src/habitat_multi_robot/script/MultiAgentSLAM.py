import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import logging
from habitat.core.simulator import Observations
from habitat.core.env import Env
import numpy as np
import random
import time

from typing import Dict

fmt = logging.Formatter('%(asctime)s %(levelname)s %(filename)s-%(funcName)s: [%(lineno)d]  %(message)s', '%Y-%m-%d %H:%M:%S')


# STOP = 0
# MOVE_FORWARD = 1
# TURN_LEFT = 2
# TURN_RIGHT = 3
# LOOK_UP = 4
# LOOK_DOWN = 5
all_actions = [1, 2, 3]
frame_rate = int(1/30. * 1000)

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def get_logger(path, clevel=logging.DEBUG, flevel=logging.DEBUG):
    _logger = logging.getLogger(path)
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        _logger.addHandler(sh)

        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        fh.setLevel(flevel)
        _logger.addHandler(fh)
    return _logger

logger = get_logger("MultiAgentSLAM.log")

def init(config="configs/tasks/MASLAM.yaml"):
    env = habitat.Env(
        config=habitat.get_config(config)
    )
    logger.debug("Environment creation successful: %s", config)
    logger.debug("Number of Agents: %d; IDs: %s", env.num_agents, env.agent_ids)
    return env

def get_all_obs(obs: Observations, env: Env):
    all_obs = {env.default_agent_id: obs}
    rest_agent_ids = list(env.agent_ids)
    rest_agent_ids.remove(env.default_agent_id)
    if rest_agent_ids:
        all_obs.update(env.get_observation_of(rest_agent_ids))
    return all_obs

def show_all_obs(all_obs: Dict, env: Env):
    rgbs = []
    states = []
    for i in env.agent_ids:
        rgbs.append(all_obs[i]["rgb"])
        states.append(env.get_state_of(i))
    rgbs = np.hstack(rgbs)
    # rgbs = np.concatenate(rgbs, axis=0)
    cv2.imshow("RGB", transform_rgb_bgr(rgbs))
    cv2.waitKey(frame_rate)

def main():
    env = init()
    agent_ids = env.agent_ids
    obs = env.reset()
    step = 0
    all_obs = get_all_obs(obs, env)
    show_all_obs(all_obs, env)

    while step < 500:
        action = {}
        for i in agent_ids:
            action[i] = random.choice(all_actions)

        obs = env.step(action)
        all_obs = get_all_obs(obs, env)
        show_all_obs(all_obs, env)
        step += 1
        time.sleep(0.05)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

