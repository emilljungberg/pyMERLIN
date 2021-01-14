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


def moco_combined(source_h5, dest_h5, reg_list):
    """Corrects a combined radial dataset from list of pickle files

    Args:
        source_h5 (str): Source file to correct
        dest_h5 (str): Output file, will first be copied from source
        reg_list (list): List of registration dictionaries
    """

    # Load data
    valid_dest_h5 = check_filename(dest_h5)

    logging.info("Copying source file")
    copyfile(source_h5, valid_dest_h5)

    logging.info("Opening %s" % valid_dest_h5)
    f = h5py.File(valid_dest_h5, 'r+')

    info = f['info'][:]
    spacing = info['voxel_size'][0]
    spokes_lo = info['spokes_lo'][0]

    n_interleaves = len(reg_list)

    traj = f['traj']
    traj_arr = traj[:]
    traj_arr_py = pyreshape(traj_arr)
    traj_arr_py_corr = traj_arr_py

    data = f['data/0000']
    data_arr = data[:]
    data_arr_py = pyreshape(data_arr)
    data_arr_py_corr = data_arr_py

    # We don't correct any lowres spokes
    idx0 = 0
    idx1 = int(spokes_lo)

    logging.info("Correcting data and trajectories")
    for (i, D_reg) in zip(range(1, n_interleaves), reg_list):
        logging.info("Processing interleave %d" % i)
        idx0 = idx1         # Start where last interleave ended
        idx1 = idx0 + D_reg['spi']

        traj_int = traj_arr_py[idx0:idx1, :, :]
        data_int = data_arr_py[idx0:idx1, :, :]

        traj_arr_py_corr[idx0:idx1, :, :] = np.matmul(traj_int, D_reg['R'])

        H = calc_H(traj_int, D_reg, spacing)
        for ircv in range(np.shape(data_arr_py)[-1]):
            data_arr_py_corr[idx0:idx1, :, ircv] = data_int[:, :, ircv]*H

    logging.info("Writing data back to H5 file")
    data[...] = pyreshape(data_arr_py_corr)
    traj[...] = pyreshape(traj_arr_py_corr)

    logging.info("Finished. Closing %s" % valid_dest_h5)
    f.close()
