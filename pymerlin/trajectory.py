import numpy as np

PHI_GOLD = np.pi*(3-np.sqrt(5))
fibonacciNum = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
                233, 377, 610, 987, 1597, 2584, 4181, 6765]


def piccini_phyllotaxis(n, nint):
    """
    Generate a spiral phyllotaxis trajectory with square root z-modulation
    according to formulation by Piccini et al. Note, this does not give a uniform
    FOV but slightly higher sampling in the x/y plane than along z.

    Args:
        n (int): Number of spokes
        nint (int): Number of interleaves

    Returns:
        [array]: Trajectory

    References:
        Piccini D, et al., Magn Reson Med. 2011;66(4):1049–56.
    """

    # Check inputs
    if n % 2:
        raise ValueError('Number of spokes must be even')

    if nint not in fibonacciNum:
        raise ValueError('Number of interleaves has to be a Fibonacci number')

    if n % nint:
        raise ValueError('Spokes per interleave must be an integer number')

    spokes_per_int = round(n/nint)
    traj_tmp = np.zeros((n, 3))
    traj = np.zeros((n, 3))

    # Calculate trajectory
    for i in range(n):
        if (i < n/2):
            theta_n = np.pi/2 * np.sqrt(1.0*i/(n/2.0))
            gz_sign = 1
        else:
            theta_n = np.pi/2 * np.sqrt((n-i)/(n/2.0))
            gz_sign = -1

        phi_n = i * PHI_GOLD
        traj_tmp[i, 0] = np.sin(theta_n) * np.cos(phi_n)
        traj_tmp[i, 1] = np.sin(theta_n) * np.sin(phi_n)
        traj_tmp[i, 2] = gz_sign * np.cos(theta_n)

    # Stack the interleaves after eachother
    for i in range(nint):
        if i % 2 == 0:
            for j in range(spokes_per_int):
                idx1 = i * spokes_per_int + j
                idx2 = i + j * nint
                traj[idx1, :] = traj_tmp[idx2, :]

        else:
            for j in range(spokes_per_int):
                idx1 = i*spokes_per_int + j
                idx2 = (n - (nint - i)) - j * nint
                traj[idx1, :] = traj_tmp[idx2, :]

    return traj


def swinbank_phyllotaxis(n, nint):
    """
    Generate a spiral phyllotaxis trajectory with cosine z-modulation
    for uniform spherical sampling.

    Args:
        n (int): Number of spokes
        nint (int): Number of interleaves

    Returns:
        [array]: Trajectory

    References: 
        Swinbank R, Purser RJ., Q J R Meteorol Soc. 2006;132(619):1769–93.
    """

    # Check inputs
    if n % 2:
        raise ValueError('Number of spokes must be even')

    if nint not in fibonacciNum:
        raise ValueError('Number of interleaves has to be a Fibonacci number')

    if n % nint:
        raise ValueError('Spokes per interleave must be an integer number')

    spokes_per_int = round(n/nint)
    traj_tmp = np.zeros((3, n))
    traj = np.zeros((3, n))

    # Calculate trajectory
    for i in range(n):

        theta_n = np.acos((n/2 - i)/(n/2))
        phi_n = i * PHI_GOLD
        traj_tmp[i, 0] = np.sin(theta_n) * np.cos(phi_n)
        traj_tmp[i, 1] = np.sin(theta_n) * np.sin(phi_n)
        traj_tmp[i, 2] = np.cos(theta_n)

    # Stack the interleaves after eachother
    for i in range(nint):
        if i % 2 == 0:
            for j in range(spokes_per_int):
                idx1 = i * spokes_per_int + j
                idx2 = i + j * nint
                traj[:, idx1] = traj_tmp[:, idx2]
        else:
            for j in range(spokes_per_int):
                idx1 = i*spokes_per_int + j
                idx2 = (n - (nint - i)) - j * nint
                traj[:, idx1] = traj_tmp[:, idx2]


def linear_phyllotaxis(n, nint, sf):
    """Isotropic Phyllotaxis trajectory with linear interleave ordering
    and arbitrary smoothness factor

    Args:
        n (int): Number of spokes
        nint (int): Number of interleaves
        sf (int): Smoothness factor

    Returns:
        array: Trajectory

    References:
        1. Swinbank R, Purser RJ., Q J R Meteorol Soc. 2006;132(619):1769–93.
        2. Piccini D, et al., Magn Reson Med. 2011;66(4):1049–56.
    """

    traj = np.zeros((n, 3))
    spi = int(n/nint)

    i = np.arange(spi)
    phi0 = i * PHI_GOLD * fibonacciNum[sf]
    z0 = 1 - 2*nint*i/(n-1)
    r = 1

    for k in range(nint):
        z = z0 - k*2/(n-1)
        phi = phi0 + k * PHI_GOLD

        theta = np.arccos(z)
        traj[k*spi: (k+1)*spi, 0] = r * np.sin(theta) * np.cos(phi)
        traj[k*spi: (k+1)*spi, 1] = r * np.sin(theta) * np.sin(phi)
        traj[k*spi: (k+1)*spi, 2] = r * z

    return traj


def wong_roos_traj(n):
    """3D Radial trajectory as formulated by Wong and Roos

    Args:
        n (int): Number of spokes

    Returns:
        array: Trajectory

    References:
        S. T. S. Wong and M. S. Roos, “A strategy for sampling on a sphere applied to 3D selective RF pulse design,” 
        Magn. Reson. Med., vol. 32, no. 6, pp. 778–784, 1994.
    """

    traj = np.zeros((n, 3))
    ni = np.arange(1, n+1)

    traj[:, 2] = (2*ni - n - 1)/n
    traj[:, 0] = np.cos(np.sqrt(n*np.pi)*np.arcsin(traj[:, 2])
                        )*np.sqrt(1-traj[:, 2]**2)
    traj[:, 1] = np.sin(np.sqrt(n*np.pi)*np.arcsin(traj[:, 2])
                        ) * np.sqrt(1-traj[:, 2]**2)

    return traj


def wong_roos_interleaved_traj(n, nint):
    """Interleaved trajectory by Wong and Roos

    Args:
        n (int): Number of spokes
        nint (int): Number of interleaves

    Returns:
        array: Trajectory

    References:
        S. T. S. Wong and M. S. Roos, “A strategy for sampling on a sphere applied to 3D selective RF pulse design,” 
        Magn. Reson. Med., vol. 32, no. 6, pp. 778–784, 1994.
    """

    traj = np.zeros((n, 3))
    spi = int(n/nint)
    ni = np.arange(1, spi+1)

    z = -(2*ni-spi-1)/spi

    ang_vel = np.sqrt(n*np.pi/nint)*np.arcsin(z)
    for m in range(nint):
        x = np.cos(ang_vel + (2*(m+1)*np.pi)/nint) * np.sqrt(1-z**2)
        y = np.sin(ang_vel + (2*(m+1)*np.pi)/nint) * np.sqrt(1-z**2)

        traj[m*spi:(m+1)*spi, 0] = x
        traj[m*spi:(m+1)*spi, 1] = y
        traj[m*spi:(m+1)*spi, 2] = z

    return traj


def traj2points(traj, npoints, OS):
    """
    Transform spoke trajectory to point trajectory

    Args:
        traj: Trajectory with shape [nspokes, 3]
        npoints: Number of readout points along spokes
        OS: Oversampling

    Returns:
        array: Trajectory with shape [nspokes, npoints, 3]
    """

    [nspokes, ndim] = np.shape(traj)

    r = (np.arange(0, npoints))/OS
    Gx, Gy, Gz = np.meshgrid(r, np.arange(nspokes), np.arange(ndim))
    traj_p = Gx*np.transpose(np.tile(traj, [npoints, 1, 1]), [1, 0, 2])

    return traj_p
