# Various helpful functions
import numpy as np


def fibonacci(k):
    """Calculate Fibonacci number

    Args:
        k (int): Which Fibonacci number

    Returns:
        int: The kth Fibonacci number
    """

    if k == 0:
        return 0
    if k == 1:
        return 1
    else:
        return fibonacci(k-1) + fibonacci(k-2)


def rotmat(rot_angles):
    """Calculate rotation matrix

    Args:
        rot_angles (array): Rotation angles (ax,ay,az)

    Returns:
        array: 3x3 rotation matrix
    """

    ax = rot_angles[0]
    ay = rot_angles[1]
    az = rot_angles[2]

    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)

    R = np.array([[cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz],
                  [cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-sx*cz],
                  [-sy,   sx*cy,          cx*cy]])

    return R
