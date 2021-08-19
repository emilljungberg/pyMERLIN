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

    xF = traj[:, :, 0]/np.max(abs(traj[:, :, 0]))/2
    yF = traj[:, :, 1]/np.max(abs(traj[:, :, 1]))/2
    zF = traj[:, :, 2]/np.max(abs(traj[:, :, 2]))/2

    H = np.exp(2j*np.pi*(xF*dx + yF*dy + zF*dz))

    return H


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
    traj_corr = np.copy(traj)

    data = f_source['noncartesian'][:]
    data_corr = np.zeros_like(data)

    logging.info("Correcting data and trajectories")
    traj_corr[:, :, :] = np.matmul(traj, reg['R'])

    H_high = calc_H(traj[spokes_lo:, :, :], reg, spacing)
    H_low = calc_H(traj[0:spokes_lo, :, :], reg, spacing/lo_scale)

    for ircv in range(np.shape(data)[-1]):
        data_corr[0, spokes_lo:, :, ircv] = data[0, spokes_lo:, :, ircv]*H_high
        data_corr[0, 0:spokes_lo, :,  ircv] = data[0,
                                                   0:spokes_lo, :, ircv]*H_low

    # Write data to destination file
    valid_dest_h5 = check_filename(dest_h5)
    logging.info("Opening destination file: %s" % valid_dest_h5)
    f_dest = h5py.File(valid_dest_h5, 'w')

    logging.info("Writing info and meta data")
    f_dest.create_dataset("info", data=info)
    f_source.copy('meta', f_dest)

    logging.info("Writing trajectory")
    traj_chunk_dims = list(traj_corr.shape)
    if traj_chunk_dims[0] > 1024:
        traj_chunk_dims[0] = 1024
    f_dest.create_dataset("trajectory", data=traj_corr,
                          chunks=tuple(traj_chunk_dims), compression='gzip')

    logging.info("Writing k-space data")
    data_chunk_dims = list(data_corr.shape)
    if data_chunk_dims[1] > 1024:
        data_chunk_dims[1] = 1024
    f_dest.create_dataset("noncartesian", dtype='c8', data=data_corr,
                          chunks=tuple(data_chunk_dims), compression='gzip')

    logging.info("Closing all files")
    f_source.close()
    f_dest.close()


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
    traj_corr = np.copy(traj)

    data = f_source['noncartesian'][0, :, :, :]
    data_corr = np.copy(data)

    # We don't correct any lowres spokes
    idx0 = 0
    idx1 = int(spokes_lo)

    logging.info("Correcting data and trajectories")
    for (i, D_reg) in enumerate(reg_list):
        logging.info("Processing interleave %d/%d" % (i+1, n_interleaves))
        idx0 = idx1         # Start where last interleave ended
        idx1 = idx0 + D_reg['spi']

        traj_int = traj[idx0:idx1, :, :]
        data_int = data[idx0:idx1, :, :]

        traj_corr[idx0:idx1, :, :] = np.matmul(traj_int, D_reg['R'])

        H = calc_H(traj_int, D_reg, spacing)
        for ircv in range(np.shape(data)[-1]):
            data_corr[idx0:idx1, :, ircv] = data_int[:, :, ircv]*H

    # Write data to destination file
    valid_dest_h5 = check_filename(dest_h5)
    logging.info("Opening destination file: %s" % valid_dest_h5)
    f_dest = h5py.File(valid_dest_h5, 'w')

    logging.info("Writing info and meta data")
    f_dest.create_dataset("info", data=info)
    f_source.copy('meta', f_dest)

    logging.info("Writing trajectory")
    f_dest.create_dataset("trajectory", data=traj_corr,
                          chunks=np.shape(traj_corr), compression='gzip')

    logging.info("Writing k-space data")
    f_dest.create_dataset("noncartesian", dtype='c8', data=data_corr[np.newaxis, ...],
                          chunks=np.shape(data_corr), compression='gzip')

    logging.info("Closing all files")
    f_source.close()
    f_dest.close()


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
    traj_corr = np.copy(traj)

    data = f_source['noncartesian'][:]
    data_corr = np.copy(data)

    # We don't correct any lowres spokes
    idx0 = 0
    idx1 = int(spokes_lo)

    logging.info("Correcting data and trajectories")
    # Loop over all segments
    for i in range(n_nav):
        logging.info("Processing segment %d/%d" % (i+1, n_nav))

        iw = i - int(np.floor(nseg/2))
        if iw < 0:
            iw = 0

        D_reg = reg_list[iw]

        idx0 = idx1                   # Start where last interleave ended
        sps = int(D_reg['spi']/nseg)  # Spokes per segment
        idx1 = idx0 + sps
        print("i0:i1: {}:{}".format(idx0, idx1))

        traj_int = traj[idx0:idx1, :, :]
        data_int = data[0, idx0:idx1, :, :]

        print("Matmul")
        traj_corr[idx0:idx1, :, :] = np.matmul(traj_int, D_reg['R'])

        H = calc_H(traj_int, D_reg, spacing)
        for ircv in range(np.shape(data)[-1]):
            data_corr[0, idx0:idx1, :, ircv] = data_int[:, :, ircv]*H

    # Write data to destination file
    valid_dest_h5 = check_filename(dest_h5)
    logging.info("Opening destination file: %s" % valid_dest_h5)
    f_dest = h5py.File(valid_dest_h5, 'w')

    logging.info("Writing info and meta data")
    try:
        f_dest.create_dataset("info", data=info)
        f_source.copy('meta', f_dest)
    except:
        print("No meta data, skipping this.")

    logging.info("Writing trajectory")
    traj_chunk_dims = list(traj_corr.shape)
    if traj_chunk_dims[0] > 1024:
        traj_chunk_dims[0] = 1024
    f_dest.create_dataset("trajectory", data=traj_corr,
                          chunks=tuple(traj_chunk_dims), compression='gzip')

    logging.info("Writing k-space data")
    data_chunk_dims = list(data_corr.shape)
    if data_chunk_dims[1] > 1024:
        data_chunk_dims[1] = 1024
    f_dest.create_dataset("noncartesian", dtype='c8', data=data_corr,
                          chunks=tuple(data_chunk_dims), compression='gzip')

    logging.info("Closing all files")
    f_source.close()
    f_dest.close()
