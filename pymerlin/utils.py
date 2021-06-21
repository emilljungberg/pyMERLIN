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


def gradient_entropy(img):
    """
    Calculates gradient entropy of ND image

    From: McGee, K.P., Manduca, A., Felmlee, J.P., Riederer, S.J. and Ehman, R.L. (2000), 
    Image metric‐based correction (Autocorrection) of motion effects: Analysis of image metrics. 
    J. Magn. Reson. Imaging, 11: 174-181. 
    https://doi.org/10.1002/(SICI)1522-2586(200002)11:2<174::AID-JMRI15>3.0.CO;2-3
    """

    h = img/np.sum(img**2)
    GE = -np.sum(h*np.log2(h))
    return GE


def parse_combreg(combreg):
    all_reg = {'rx': [], 'ry': [], 'rz': [], 'dx': [], 'dy': [], 'dz': []}
    for k in all_reg.keys():
        for i in range(len(combreg)):
            all_reg[k].append(combreg[i][k])

    return all_reg
