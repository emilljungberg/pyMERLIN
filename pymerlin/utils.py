# -*- coding: utf-8 -*-
"""
Utilities for MERLIN
"""

import numpy as np


def fibonacci(k):
    """Calculate Fibonacci number

    Args:
        k (int): Which Fibonacci number

    Returns:
        int: The kth Fibonacci number
    """

    if type(k) != int:
        raise TypeError("k needs to be an integer")
    if k < 0:
        raise ValueError("k needs to be positive")
    if k == 0:
        return 0
    if k == 1:
        return 1
    else:
        return fibonacci(k-1) + fibonacci(k-2)


def rotmat(rot_angles):
    """Calculate rotation matrix

    https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions

    Args:
        rot_angles (array): Rotation angles (roll, pitch, yaw / x,y,z)

    Returns:
        array: 3x3 rotation matrix
    """

    alpha = rot_angles[2]
    beta = rot_angles[1]
    gamma = rot_angles[0]

    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)

    R = np.array([[ca*cb, ca*sb*sg-sa*cg, ca*sb*cg+sa*sg],
                  [sa*cb, sa*sb*sg+ca*cg, cg*sb*sb-sg*ca],
                  [-sb,   sg*cb,          cb*cg]])

    return R


def rotmat_versor(versor):
    """
    Calculates rotation matrix based on a versor. If length 3, assuming only vector part and will calculate
    4th element to make magnitude 1.

    Args:
        versor (array): 3 or 4 element versor

    Returns:
        array: 3x3 rotation matrix
    """

    if len(versor) == 4:
        q0, q1, q2, q3 = versor

    elif len(versor) == 3:
        q1, q2, q3 = versor
        q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
    else:
        return TypeError("Versor must be of lenfth 3 or 4")

    R = np.array([[1-2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q0*q2+q1*q3)],
                  [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2), 2*(q0*q1+q2*q3), q0**2-q1**2-q2**2+q3**2]])

    return R


def versor_to_euler(versor):
    """
    Calculates the intrinsic euler angles from a 3 or 4 element versor

    Args:
        versor (array): 3 or 4 element versor

    Returns:
        array: rotation angles (rx, ry, rz)
    """

    if len(versor) == 4:
        q0, q1, q2, q3 = versor

    elif len(versor) == 3:
        q1, q2, q3 = versor
        q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
    else:
        return TypeError("Versor must be of lenfth 3 or 4")

    rz = np.arctan2(2*(q0*q1+q2*q3), (1-2*(q1**2+q2**2)))
    ry = np.arcsin(2*(q0*q2 - q3*q1))
    rx = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))

    return rx, ry, rz


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
