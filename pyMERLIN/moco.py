import h5py
import logging
from shutil import copyfile
import pickle
from .dataIO import *


def calc_H(traj, D, spacing):
    """
    Calculate phase correction matrix

    Inputs:
        traj: trajectory
        D: Dictionary with correction factors
        spacing: Voxel spacing in data to be corrected
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
    """
    Reshape radial between python and riesling format
    """

    return np.reshape(np.reshape(arr, [1, np.prod(np.shape(arr))]), np.shape(arr)[::-1])


def moco_interleave(source_h5, dest_h5, corr_pickle):
    """
    Corrects h5 file based on correction factors in pickle file

    Inputs:
        - source_h5: Source file to correct
        - dest_h5: Output file, will first be copied from source
        - corr_pickle: Correction factors in pickle file
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


def moco_combined(source_h5, dest_h5, corr_pickle_list, spi, discard_ints=[]):
    """
    Corrects a combined radial dataset from list of pickle files

    Inputs:
        - source_h5: Source file to correct
        - dest_h5: Output file, will first be copied from source
        - corr_pickle_list: List pickle files with correction factors
        - spi: Spokes per interleave (assuming fixed width now)
        - discard_ints: Interleaves to discard, i.e. set to 0.
    """

    # Load data
    valid_dest_h5 = check_filename(dest_h5)

    logging.info("Copying source file")
    copyfile(source_h5, valid_dest_h5)

    logging.info("Opening %s" % valid_dest_h5)
    f = h5py.File(valid_dest_h5, 'r+')

    info = f['info'][:]
    spacing = info['voxel_size'][0]
    spokes_hi = info['spokes_hi'][0]
    spokes_lo = info['spokes_lo'][0]

    n_pickle = np.size(corr_pickle_list)
    n_interleaves = int(spokes_hi/spi)

    traj = f['traj']
    traj_arr = traj[:]
    traj_arr_py = pyreshape(traj_arr)
    traj_arr_py_corr = traj_arr_py

    data = f['data/0000']
    data_arr = data[:]
    data_arr_py = pyreshape(data_arr)
    data_arr_py_corr = data_arr_py

    # We assume that we don't correct the first (0th) intereave
    logging.info("Correcting data and trajectories")
    for (i, pfile) in zip(range(1, n_interleaves), corr_pickle_list):
        if i in discard_ints:
            data_arr_py_corr[idx0:idx1, :, :] = 0
        else:
            logging.info("Processing %s" % pfile)
            idx0 = spokes_lo + i*spi    # Assuming first interleave is the reference
            idx1 = idx0 + spi

            traj_int = traj_arr_py[idx0:idx1, :, :]
            data_int = data_arr_py[idx0:idx1, :, :]

            D_reg = pickle.load(open(pfile, 'rb'))

            traj_arr_py_corr[idx0:idx1, :, :] = np.matmul(traj_int, D_reg['R'])

            H = calc_H(traj_int, D_reg, spacing)
            for ircv in range(np.shape(data_arr_py)[-1]):
                data_arr_py_corr[idx0:idx1, :, ircv] = data_int[:, :, ircv]*H

    logging.info("Writing data back to H5 file")
    data[...] = pyreshape(data_arr_py_corr)
    traj[...] = pyreshape(traj_arr_py_corr)

    logging.info("Finished. Closing %s" % valid_dest_h5)
    f.close()
