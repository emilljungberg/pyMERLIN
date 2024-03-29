# -*- coding: utf-8 -*-
"""
MERLIN includes several convenience functions for data input and output. 
Since pyMERLIN is tightly integrated with RIESLING, functions for reading k-space and image ``.h5`` files have been included.
"""

import argparse
import logging
import os

import h5py
import ismrmrd
import itk
import numpy as np
import xmltodict


def parse_fname(fname):
    """Parse the filename from a NIFTI file

    Args:
        fname (str): Input filename

    Returns:
        str: Filename witout file ending
    """
    bname, ext = os.path.splitext(os.path.basename(fname))
    if ext.lower() == '.nii':
        return bname
    elif ext.lower() == '.gz':
        bname2, ext2 = os.path.splitext(bname)
        if ext2 == '.nii':
            return bname2


def arg_check_h5(fname):
    """Check h5 file ending

    Args:
        fname (str): Filename to check

    Returns:
        str: Filename or error
    """

    bname, ext = os.path.splitext(fname)
    if ext.lower() in ['.h5', '.hd5', '.hf5', '.hdf5']:
        return fname
    else:
        raise ValueError(
            "{} doesn't seem to have the right file ending (HDF5 file)".format(fname))


def arg_check_nii(fname):
    """Check nifti file ending

    Args:
        fname (str): Filename to check

    Returns:
        str: Filename or error
    """

    bname, ext = os.path.splitext(fname)
    if ext.lower() == '.nii':
        return fname
    elif ext.lower() == '.gz':
        bname2, ext2 = os.path.splitext(ext)
        if ext2 == '.nii':
            return fname

    return argparse.ArgumentError("{} doesn't seem to be a valid nifti file".format(fname))


def check_filename(fname):
    """
    Check if file exists

    If file already exist it will append a number to produce a unique filename.


    Args:
        fname (str): Filename

    Returns:
        str: Unique filename 
    """

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
    """Read ISMRM Raw Data format

    Args:
        fname (str): ISMRM Raw Data filename

    Returns:
        dict: Data and meta data as dictionary
    """

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
    """Create an ITK image object from array

    Will take magnitude of the data. Does not impose any geometry.

    Args:
        img_array (array): 3D image array 
        spacing (list): Voxel size (dx,dy,dz)
        corners (list, optional): Not implemented. Defaults to None.
        max_image_value (float, optional): Rescale data to max value. Defaults to None.
        dtype (itk.dtype, optional): ITK data type for casting. Defaults to None.

    Returns:
        itk.Image: ITK Image object
    """

    # Can only do single volume
    img = itk.image_from_array(np.abs(np.ascontiguousarray(img_array)))

    # Don't assume isotropic voxels
    img.SetSpacing([float(x) for x in spacing])

    # Set origin in center of the image
    img.SetOrigin([-img_array.shape[i]/2*spacing[i] for i in range(3)])

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


def read_image_h5(h5_file, vol=0, echo=0):
    """Read image h5 file

    Args:
        h5_file (str): File name

    Returns:
        (array,array): Image array and voxel spacing
    """

    f = h5py.File(h5_file, 'r')

    data = f['image'][echo, :, :, :, vol]
    spacing = f['info'][0][1]

    f.close()

    return data, spacing


def read_radial_h5(h5_file):
    """Read radial k-space h5 file

    Args:
        h5_file (str): Filename

    Returns:
        (array,array): k-space data and trajectory
    """

    f = h5py.File(h5_file, 'r')

    # Reshape trajectory
    traj = f['trajectory']

    # Reshape data
    data = f['noncartesian']

    return data, traj


def make_3D(img):
    """Extract first 3D volume from 4D dataset 

    Args:
        img (np.array): 4D or 3D array

    Returns:
        np.array: 3D array
    """

    if len(img.shape) > 3:
        return img[:, :, :, 0]

    return img


def write_kspace_h5(f_dest, noncartesian, trajectory, info, f_source=None):
    f_dest = h5py.File(f_dest, 'w')

    logging.info("Writing info and meta data")
    f_dest.create_dataset("info", data=info)

    if f_source:
        try:
            f_source.copy('meta', f_dest)
        except:
            logging.info("No meta data")
        f_source.copy('echoes', f_dest)

    logging.info("Writing trajectory")
    traj_chunk_dims = list(trajectory.shape)
    if traj_chunk_dims[0] > 1024:
        traj_chunk_dims[0] = 1024

    f_dest.create_dataset("trajectory", data=trajectory,
                          chunks=tuple(traj_chunk_dims), compression='gzip')

    logging.info("Writing k-space data")
    data_chunk_dims = list(noncartesian.shape)
    if data_chunk_dims[1] > 1024:
        data_chunk_dims[1] = 1024
    f_dest.create_dataset("noncartesian", dtype='c8', data=noncartesian,
                          chunks=tuple(data_chunk_dims), compression='gzip')

    logging.info("Closing {}".format(f_dest))
    f_dest.close()


def get_merlin_fields():
    valid_input_args = ['nspokes',
                        'nlores',
                        'spoke_step',
                        'make_brain_mask',
                        'brain_mask_file',
                        'sense_maps',
                        'cg_its',
                        'ds',
                        'fov',
                        'overwrite_files',
                        'riesling_verbosity',
                        'ref_nav_num',
                        'metric',
                        'batch_itk',
                        'batch_riesling',
                        'threads_itk',
                        'threads_riesling']

    return valid_input_args


def check_merlin_inputs(fname):
    valid_input_args = get_merlin_fields()
    with open('merlin_input_args.txt') as f:
        eof = False
        input_arguments = {}
        while not eof:
            line = f.readline().split('\n')[0]
            if len(line) > 0:
                arg = line.split('=')[0]
                val = line.split('=')[1]
                if arg not in input_args:
                    raise ValueError('%s is not a valid input argument' % arg)
                input_arguments[arg] = val
            else:
                eof = True

    # Check for invalid combinations
    if (input_arguments['make_brain_mask'] == 1) and (len(input_arguments['brain_mask_file']) > 0):
        raise ValueError(
            'Cannot use make_brain_mask and brain_mask_file together. Choose one')
