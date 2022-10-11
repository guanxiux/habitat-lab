
from tf.transformations import quaternion_from_euler, euler_from_quaternion, _AXES2TUPLE

import quaternion
import numpy as np

for _ in range(100):
    euler = np.random.sample(3) * np.pi * 2
    q1 = quaternion.from_euler_angles(*euler).components
    for key in _AXES2TUPLE.keys():
        q2 = quaternion_from_euler(*euler, key)
        temp = list(q2)
        q3 = [temp[3]] + temp[0:3]
        if np.allclose(q1, q2):
            print(key)
        if np.allclose(q1, q3):
            print(key)
            print("FUCK")