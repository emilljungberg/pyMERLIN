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


def parse_combreg(combreg):
    """Parse combined registration object

    Args:
        combreg (dict): Dictionary with registration results

    Returns:
        dict: Parsed registration results
    """
    all_reg = {'rx': [], 'ry': [], 'rz': [], 'dx': [], 'dy': [], 'dz': []}
    for k in all_reg.keys():
        for i in range(len(combreg)):
            all_reg[k].append(combreg[i][k])

    return all_reg


def make_tukey(n, a=0.5):
    """Make a tukey window

    Args:
        n (int): Number of points
        a (float, optional): Width of window. Defaults to 0.5.

    Returns:
        np.array: Weights
    """
    x = np.arange(n)
    weights = np.ones_like(x, dtype=float)
    weights[0:int(a*n/2)] = 1/2*(1-np.cos(2*np.pi*x[0:int(a*n/2)]/(a*n)))
    weights[-int(a*n/2):] = weights[0:int(a*n/2)][::-1]

    return weights
