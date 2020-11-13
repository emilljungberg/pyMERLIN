import ismrmrd
import h5py
import xmltodict
import numpy as np
import itk
from shutil import copyfile
import logging
import os


def check_filename(fname):
    if os.path.exists(fname):
        logging.warning("%s already exists" % fname)
        fpath, fext = os.path.splitext(fname)
        counter = 1
        fname = "%s_%d%s" % (fpath, counter, fext)
        while os.path.exists(fname):
            counter += 1

    logging.info("%s does not exists" % fname)
    return fname


def read_ismrmrd(fname):

    # Read in dataset as ISMRM Raw Data file to get header
    f_ismrmrd = ismrmrd.Dataset(fname, '/dataset', True)
    header = xmltodict.parse(f_ismrmrd.read_xml_header())

    user_vars = header['ismrmrdHeader']['userParameters']['userParameterDouble']
    for l in user_vars:
        vals = l.values()
        if 'Highres Spokes' in vals:
            nSpokesHigh = int(list(vals)[1])
        if 'WASPI Spokes' in vals:
            nSpokesLow = int(list(vals)[1])

    acq = f_ismrmrd.read_acquisition(0)
    nrcv = acq.active_channels
    npts = acq.number_of_samples

    # Now read it in as a native HDF5 file for faster data reading
    fh5 = h5py.File(fname, 'r')
    dataset = fh5['dataset']
    data = np.asarray(dataset['data'])

    # Load highres data
    traj = np.zeros((nSpokesHigh, npts, 3))
    traj_low = None
    ks = np.zeros((nrcv, nSpokesHigh, npts), dtype=np.complex64)
    ks_low = None
    for i in range(nSpokesHigh):
        ks[:, i, :] = data[i][2].view(np.complex64).reshape((nrcv, npts))
        traj[i, :, :] = data[i][1].reshape((npts, 3))

    # Load lowres data
    if nSpokesLow > 0:
        traj_low = np.zeros((nSpokesLow, npts, 3))
        ks_low = np.zeros((nrcv, nSpokesLow, npts), dtype=np.complex64)
        for i in range(nSpokesLow):
            ks_low[:, i, :] = data[nSpokesHigh +
                                   i][2].view(np.complex64).reshape((nrcv, npts))
            traj_low[i, :, :] = data[nSpokesHigh + i][1].reshape((npts, 3))

    D = {'RawHigh': ks, 'RawLow': ks_low, 'TrajHigh': traj,
         'TrajLow': traj_low, 'Header': header}

    # Close files
    fh5.close()
    f_ismrmrd.close()

    return D


def create_image(img_array, spacing, corners=None, max_image_value=None, dtype=None):
    """
    Creates an itk image object from the img_array. Uses information for
    spacing to set the voxel size and the pfile corner information to get the
    correct orientation.

    Inputs:
        img_array: Image array as numpy matrix
        spacing: Voxel size (dx,dy,dz)
        corners: NOT IMPLEMENTED

    Outputs:
        img: ITK image object
    """

    # Can only do single volume
    img = itk.image_from_array(np.abs(np.ascontiguousarray(img_array)))

    # Don't assume isotropic voxels
    img.SetSpacing([float(x) for x in spacing])

    if max_image_value:
        # Rescale image intensity
        RescaleFilterType = itk.RescaleIntensityImageFilter[type(
            img), type(img)]
        rescaleFilter = RescaleFilterType.New()
        rescaleFilter.SetInput(img)
        rescaleFilter.SetOutputMinimum(0)
        rescaleFilter.SetOutputMaximum(max_image_value)
        rescaleFilter.Update()
        img_out = rescaleFilter.GetOutput()

    else:
        img_out = img

    # Cast filter
    if dtype:
        InputPixelType = itk.template(img_out)[1][0]
        InputImageType = itk.Image[InputPixelType, 3]
        OutputImageType = itk.Image[dtype, 3]

        cast_filter = itk.CastImageFilter[InputImageType, OutputImageType].New(
        )
        cast_filter.SetInput(img_out)
        cast_filter.Update()
        img_out = cast_filter.GetOutput()

    return img_out


def read_image_h5(h5_file):
    f = h5py.File(h5_file, 'r')

    data = f['data/0000'][:]
    spacing = f['info'][0][1]

    f.close()

    return data, spacing


def read_radial_h5(h5_file):
    """
    Read riesling h5 file and return python style data and trajectory arrays
    """

    f = h5py.File(h5_file, 'r')

    # Reshape trajectory
    traj = f['traj']
    traj_flat = np.reshape(traj, (1, np.prod(traj.shape)))
    traj_rs = np.reshape(
        traj_flat, [traj.shape[2], traj.shape[1], traj.shape[0]])

    # Reshape data
    data = f['data/0000']
    data_rs = np.reshape(np.reshape(data, (1, np.prod(data.shape))), [
                         data.shape[2], data.shape[1], data.shape[0]])

    return data_rs, traj_rs


def modify_h5(source_h5, dest_h5, data, traj):

    try:
        os.remove(dest_h5)
        print("Removing old file")
    except:
        pass

    copyfile(source_h5, dest_h5)

    f = h5py.File(dest_h5, 'r+')
    moco_traj = f['traj']
    moco_data = f['data/0000']

    traj_rs = np.reshape(np.reshape(
        traj, [1, np.prod(np.shape(traj))]), np.shape(traj)[::-1])
    data_rs = np.reshape(np.reshape(
        data, [1, np.prod(np.shape(data))]), np.shape(data)[::-1])

    moco_data[...] = data_rs
    moco_traj[...] = traj_rs

    f.close()

    return dest_h5
