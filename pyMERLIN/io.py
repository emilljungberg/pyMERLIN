import ismrmrd
import h5py
import xmltodict
import numpy as np
import itk


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


def create_image(img_array, spacing, corners=None, max_image_value=1000):
    """
    Creates an itk image object from the img_array. Uses information for
    spacing to set the voxel size and the pfile corner information to get the
    correct orientation.

    Inputs:
        img_array: Image array as numpy matrix
        spacing: Voxel size
        corners: NOT IMPLEMENTED

    Outputs:
        img: ITK image object
    """

    img = itk.image_from_array(img_array)

    # Assume isotropic voxels
    img.SetSpacing([spacing, spacing, spacing])

    # Rescale image intensity
    RescaleFilterType = itk.RescaleIntensityImageFilter[type(img), type(img)]
    rescaleFilter = RescaleFilterType.New()
    rescaleFilter.SetInput(img)
    rescaleFilter.SetOutputMinimum(0)
    rescaleFilter.SetOutputMaximum(max_image_value)
    rescaleFilter.Update()

    return rescaleFilter.GetOutput()
