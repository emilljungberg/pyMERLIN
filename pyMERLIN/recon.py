import numpy as np
import sigpy

PHI_GOLD = np.pi*(3-np.sqrt(5))
fibbonaciNum = [3, 5, 8, 13, 21, 34, 55, 89, 144,
                233, 377, 610, 987, 1597, 2584, 4181, 6765]


def traj_phyllotaxis_sqrt(n, nint):
    """
    Generate a spiral phyllotaxis trajectory with square root z-modulation
    according to formulation by Piccini et al.
    Note, this does not give a uniform FOV but slightly higher sampling
    in the x/y plane than along z

    Input:
        n: Number of spokes
        nint: Number of interleaves (Fibonacci number)

    Output:
        traj: Trajectory with interleaves stacked

    Ref: Piccini D, et al., Magn Reson Med. 2011;66(4):1049–56.
    """

    # Check inputs
    if n % 2:
        raise ValueError('Number of spokes must be even')

    if nint not in fibbonaciNum:
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


def traj_phyllotaxis_acos(n, nint):
    """
    Generate a spiral phyllotaxis trajectory with cosine  z-modulation
    for uniform spherical sampling.

    Ref: Swinbank R, Purser RJ., Q J R Meteorol Soc. 2006;132(619):1769–93.
    """

    # Check inputs
    if n % 2:
        raise ValueError('Number of spokes must be even')

    if nint not in fibbonaciNum:
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


def traj2points(traj, npoints, OS):
    """
    Transform spoke trajectory to point trajectory

    Input:
        traj: Trajectory with shape [nspokes, 3]
        npoints: Number of readout points along spokes
        OS: Oversampling

    Output:
        trajp: Trajectory with shape [nspokes, npoints, 3]
    """

    [nspokes, ndim] = np.shape(traj)

    r = (np.arange(0, npoints))/OS
    Gx, Gy, Gz = np.meshgrid(r, np.arange(nspokes), np.arange(ndim))
    traj_p = Gx*np.transpose(np.tile(traj, [npoints, 1, 1]), [1, 0, 2])

    return traj_p


def dc_filter(n):
    """
    Simple r^2 DC filter for 3D radial acquisition
    """

    r = np.linspace(0, 1, num=n)
    dcf = r**2

    return dcf


def fermi_filter(n, rf, wf):
    """
    Fermi filter for radial out data

    Inputs:
        - n: Number of points along spoke
        - rf: Filter radius
        - wf: Filter width

    Outputs:
        - filt: 1D Fermi filter
    """
    r = np.linspace(0, 1, num=n)
    filt = 1.0/(1+np.exp((r-rf)/wf))

    return filt


def sense_selfcalib(y, coord, oshape=None, rf=0.25, wf=0.05):
    """
    Generate SENSE maps from 3D radial data using the center of k-space.
    Applies a fermi filter to lowpass filter the data. DC filter is 
    applied in here as well.

    Based on: McKenzie CA et al. Magn Reson Med. 2002;47(3):529–38. 

    Inputs:
        - y: Radial k-space data (raw)
        - coord: Coordinates/trajectory point by point
        - oshape: Shape of output data
        - rf: Radius for fermi filter (0.25)
        - wf: Width for fermi filter (0.05)

    Returns:
        - SENSE: Sensitivity maps
    """

    [nrcv, nspokes, npts] = np.shape(y)
    dcf = dc_filter(npts)
    ff = fermi_filter(npts, rf, wf)

    # Calculate low resolution coil images
    I_coils = sigpy.nufft_adjoint(y*ff*dcf, coord, oshape=oshape)

    # Calculate SoS image
    I_rss = np.sum(np.abs(I_coils)**2, 0)**0.5

    # Make SENSE maps by dividing out PD information from the coil images
    SENSE = I_coils/I_rss

    return SENSE
