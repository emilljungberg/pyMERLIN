# -*- coding: utf-8 -*-
"""
Tools for applying motion correction to k-space data
"""

import logging

import h5py

from .dataIO import *


def calc_H(traj, D, spacing):
    """Calculate phase correction matrix

    Args:
        traj (array): Trajectory to correct
        D (dict): Correction factors ({'dx','dy', 'dz'})
        spacing (array): Voxel spacing [x,y,z]

    Returns:
        array: Phase correction matrix
    """

    dx = D['dx']/spacing[0]
    dy = D['dy']/spacing[1]
    dz = D['dz']/spacing[2]

    # We assume that the trajectory is normalised to 1
    xF = traj[:, :, 0]/2
    yF = traj[:, :, 1]/2
    zF = traj[:, :, 2]/2

    H = np.exp(2j*np.pi*(xF*dx + yF*dy + zF*dz))

    return H


def apply_moco(data_in, traj_in, D_reg, spacing):
    """
    ITK versor applies the rotation and then the translation
    Correct approach is thus to first rotate trajectory then apply
    the corresponding phase correcttion on these coordinates
    """

    traj_corr = np.matmul(traj_in, D_reg['R'])
    H = calc_H(traj_corr, D_reg, spacing)

    if np.isnan(H[:]).any():
        raise ValueError("H is nan")

    data_corr = np.zeros_like(data_in)
    for ircv in range(np.shape(data_in)[-1]):
        data_corr[..., ircv] = data_in[..., ircv]*H

    return data_corr, traj_corr


def moco_single(source_h5, dest_h5, reg):
    """Corrects a radial dataset from a pickle file.

    Args:
        source_h5 (str): Source file to correct
        dest_h5 (str): Output file
        reg_list (list): Reg file
    """

    # Load data
    logging.info("Opening source file: %s" % source_h5)
    f_source = h5py.File(source_h5, 'r')

    info = f_source['info'][:]
    spacing = info['voxel_size'][0]
    spokes_lo = info['spokes_lo'][0]
    lo_scale = info['lo_scale'][0]

    traj = f_source['trajectory'][:]
    traj_corr = np.zeros_like(traj)

    data = f_source['noncartesian'][:]
    data_corr = np.zeros_like(data)

    logging.info("Correcting data and trajectories")
    traj_corr = np.matmul(traj, reg['R'])

    H_high = calc_H(traj_corr[spokes_lo:, :, :], reg, spacing)
    H_low = calc_H(traj_corr[0:spokes_lo, :, :], reg, spacing/lo_scale)

    for ircv in range(np.shape(data)[-1]):
        data_corr[0, spokes_lo:, :, ircv] = data[0, spokes_lo:, :, ircv]*H_high
        data_corr[0, 0:spokes_lo, :,  ircv] = data[0,
                                                   0:spokes_lo, :, ircv]*H_low

    # Write data to destination file
    valid_dest_h5 = check_filename(dest_h5)
    logging.info("Opening destination file: %s" % valid_dest_h5)
    write_kspace_h5(valid_dest_h5, data_corr,
                    traj_corr, info, f_source=f_source)
    f_source.close()


def moco_combined(source_h5, dest_h5, reg_list):
    """Corrects a combined radial dataset from list of pickle files

    Args:
        source_h5 (str): Source file to correct
        dest_h5 (str): Output file
        reg_list (list): List of registration dictionaries
    """

    # Load data
    logging.info("Opening source file: %s" % source_h5)
    f_source = h5py.File(source_h5, 'r')

    info = f_source['info'][:]
    spacing = info['voxel_size'][0]
    spokes_lo = info['spokes_lo'][0]

    n_interleaves = len(reg_list)

    traj = f_source['trajectory'][:]
    traj_corr = np.zeros_like(traj)

    data = f_source['noncartesian'][:]
    data_corr = np.zeros_like(data)

    # We don't correct any lowres spokes
    idx0 = 0
    idx1 = int(spokes_lo)

    if spokes_lo > 0:
        data_corr[0, idx0:idx1, :, :] = data[0, idx0:idx1, :, :]
        traj_corr[idx0:idx1, :, :] = traj[idx0:idx1, :, :]

    logging.info("Correcting data and trajectories")
    for (i, D_reg) in enumerate(reg_list):
        logging.info("Processing interleave %d/%d" % (i+1, n_interleaves))
        idx0 = idx1
        idx1 = idx0 + D_reg['spi']

        tc, dc = apply_moco(data_in=data[0, idx0:idx1, :, :], traj_in=traj[idx0:idx1, :, :],
                            D_reg=D_reg, spacing=spacing)

        data_corr[0, idx0:idx1, ...] = dc
        traj_corr[idx0:idx1, ...] = tc

    # Write data to destination file
    valid_dest_h5 = check_filename(dest_h5)
    logging.info("Opening destination file: %s" % valid_dest_h5)
    write_kspace_h5(valid_dest_h5, data_corr,
                    traj_corr, info, f_source=f_source)

    logging.info("Closing all files")
    f_source.close()


def moco_sw(source_h5, dest_h5, reg_list, nseg):
    """Corrects radial dataset from a sliding window reconstruction.

    Args:
        source_h5 (str): Source file to correct
        dest_h5 (str): Output file
        reg_list (list): List of registration dictionaries
        nseg (int): Number of segments per interleave (equivalent to sliding window step)
    """

    # Load data
    logging.info("Opening source file: %s" % source_h5)
    f_source = h5py.File(source_h5, 'r')

    info = f_source['info'][:]
    spacing = info['voxel_size'][0]
    spokes_lo = info['spokes_lo'][0]

    n_nav = len(reg_list)

    traj = f_source['trajectory'][:]
    traj_corr = np.zeros_like(traj)

    data = f_source['noncartesian'][:]
    data_corr = np.zeros_like(data)

    # We don't correct any lowres spokes
    idx0 = 0
    idx1 = int(spokes_lo)

    if spokes_lo > 0:
        data_corr[0, idx0:idx1, :, :] = data[0, idx0:idx1, :, :]
        traj_corr[idx0:idx1, :, :] = traj[idx0:idx1, :, :]

    logging.info("Correcting data and trajectories")

    # Loop over all segments
    for iseg in range(n_nav+nseg-1):

        inav = iseg - int(np.floor(nseg/2))
        if inav < 0:
            inav = 0
        if inav > n_nav-1:
            inav = n_nav-1

        # Match segment to navigator for recon params
        logging.info("Processing segment %d/%d Using Nav %d" %
                     (iseg+1, n_nav, inav))

        D_reg = reg_list[inav]

        idx0 = idx1                   # Start where last interleave ended
        sps = int(D_reg['spi']/nseg)  # Spokes per segment
        idx1 = idx0 + sps
        print("i0:i1: {}:{}".format(idx0, idx1))

        dc, tc = apply_moco(data_in=data[0, idx0:idx1, :, :], traj_in=traj[idx0:idx1, :, :],
                            D_reg=D_reg, spacing=spacing)

        data_corr[0, idx0:idx1, ...] = dc
        traj_corr[idx0:idx1, ...] = tc

    # Write data to destination file
    valid_dest_h5 = check_filename(dest_h5)
    logging.info("Opening destination file: %s" % valid_dest_h5)
    write_kspace_h5(valid_dest_h5, data_corr,
                    traj_corr, info, f_source=f_source)

    logging.info("Closing all files")
    f_source.close()
