
from tf.transformations import quaternion_from_euler, euler_from_quaternion, _AXES2TUPLE

import quaternion
import numpy as np
from numpy import pi
from habitat.utils.geometry_utils import quaternion_to_list

for key in _AXES2TUPLE.keys():
    fail = False
    for i in range(20):
        for j in range(20):
            for k in range(20):
                euler = [i*pi/10, j*pi/10, k*pi/10]
                q1 = quaternion.from_euler_angles(*euler)
                q1 = quaternion_to_list(q1)
                q2 = quaternion_from_euler(*euler, key)
                if not np.allclose(q1, q2):
                    fail = True
                    break
            if fail:
                break
        if fail:
            break
    if not fail:
        print(key)
        print("FUCK")