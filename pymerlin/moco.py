import h5py
import logging
from shutil import copyfile
import pickle
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

    xF = traj[:, :, 0]/np.max(traj[:, :, 0])/2
    yF = traj[:, :, 1]/np.max(traj[:, :, 1])/2
    zF = traj[:, :, 2]/np.max(traj[:, :, 2])/2

    H = np.exp(2j*np.pi*(xF*dx + yF*dy + zF*dz))

    return H


def pyreshape(arr):
    """Reshapes data for riesling

    Args:
        arr (array): Input array

    Returns:
        array: Reformatted
    """

    return np.reshape(np.reshape(arr, [1, np.prod(np.shape(arr))]), np.shape(arr)[::-1])


def moco_interleave(source_h5, dest_h5, corr_pickle):
    """Apply motion correctionn to h5 interleave

    Args:
        source_h5 (str): Source .h5 file to correct
        dest_h5 (str): Output filename for corrected .h5 file
        corr_pickle (str): Pickle file with correction factors
    """
    valid_dest_h5 = check_filename(dest_h5)

    logging.info("Copying source file")
    copyfile(source_h5, valid_dest_h5)

    logging.info("Opening %s" % valid_dest_h5)
    f = h5py.File(valid_dest_h5, 'r+')
    info = f['info'][:]
    spacing = info['voxel_size'][0]
    D_reg = pickle.load(open(corr_pickle, 'rb'))

    logging.info("Correcting trajectory and data")
    # Correct trajectory
    traj = f['traj']
    traj_arr = traj[:]
    traj_arr_py = pyreshape(traj_arr)
    traj_arr_py_corr = np.matmul(traj_arr_py, D_reg['R'])

    # Correct data
    data = f['data/0000']
    data_arr = data[:]
    data_arr_py = pyreshape(data_arr)
    data_arr_py_corr = np.zeros_like(data_arr_py)
    H = calc_H(traj_arr_py, D_reg, spacing)

    for ircv in range(np.shape(data_arr_py)[-1]):
        data_arr_py_corr[:, :, ircv] = data_arr_py[:, :, ircv]*H

    # Write data back to H5 file
    logging.info("Writing back corrected data")
    data[...] = pyreshape(data_arr_py_corr)
    traj[...] = pyreshape(traj_arr_py_corr)

    logging.info("Closing %s" % valid_dest_h5)
    f.close()


def moco_single(source_h5, dest_h5, reg):
    """Corrects a radial dataset from a pickle file

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

    traj = f_source['traj'][:]
    traj_corr = np.copy(traj)

    data = f_source['volumes/0000'][:]
    data_corr = np.zeros_like(data)

    logging.info("Correcting data and trajectories")
    traj_corr[:, :, :] = np.matmul(traj, reg['R'])

    H_high = calc_H(traj[spokes_lo:, :, :], reg, spacing)
    H_low = calc_H(traj[0:spokes_lo, :, :], reg, spacing/lo_scale)

    for ircv in range(np.shape(data)[-1]):
        data_corr[spokes_lo:, :, ircv] = data[spokes_lo:, :, ircv]*H_high
        data_corr[0: spokes_lo, :,  ircv] = data[0:spokes_lo, :, ircv]*H_low

    # Write data to destination file
    valid_dest_h5 = check_filename(dest_h5)
    logging.info("Opening destination file: %s" % valid_dest_h5)
    f_dest = h5py.File(valid_dest_h5, 'w')

    logging.info("Writing info and meta data")
    f_dest.create_dataset("info", data=info)
    f_source.copy('meta', f_dest)

    logging.info("Writing trajectory")
    f_dest.create_dataset("traj", data=traj_corr,
                          chunks=np.shape(traj_corr), compression='gzip')

    logging.info("Writing k-space data")
    data_grp = f_dest.create_group("volumes")
    data_grp.create_dataset("0000", dtype='c8', data=data_corr,
                            chunks=np.shape(data_corr), compression='gzip')

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

    traj = pyreshape(f_source['traj'][:])
    traj_corr = np.copy(traj)

    data = pyreshape(f_source['volumes/0000'][:])
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
    traj_out = pyreshape(traj_corr)
    f_dest.create_dataset("traj", data=traj_out,
                          chunks=np.shape(traj_out), compression='gzip')

    logging.info("Writing k-space data")
    data_out = pyreshape(data_corr)
    data_grp = f_dest.create_group("volumes")
    data_grp.create_dataset("0000", dtype='c8', data=data_out,
                            chunks=np.shape(data_out), compression='gzip')

    logging.info("Closing all files")
    f_source.close()
    f_dest.close()
