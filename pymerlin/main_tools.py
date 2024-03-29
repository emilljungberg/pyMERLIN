#!/usr/bin/env python3
from .plot import plot_3plane
from .dataIO import read_image_h5, parse_fname
import matplotlib.pyplot as plt
import argparse
import os

import h5py
import numpy as np
import SimpleITK as sitk


def create_info(matrix, voxel_size, read_points, read_gap, spokes_hi, spokes_lo, lo_scale,
                channels, volumes, tr=0, origin=[0, 0, 0], direction=None):
    """
    Creates a numpy structured array for riesling h5 files.

    Inputs:
        - matrix: Matrix size (x,y,z)
        - voxel_size: Voxel size in mm (x,y,z)
        - read_points: Number of readout points along the spoke
        - read_gap: Deadtime gap
        - spokes_hi: Number of highres spokes
        - spokes_lo: Number of lowres spokes
        - lo_scale: Scale factor of the low res spokes
        - channels: Number of receive channels
        - volumes: Number of volumes
        - tr: Repetition time (Default=0)
        - origin: Origin of image (x,y,z) (Default: 0,0,0)
        - direction: Orientation matrix (Default: eye)

    Return: D (structured numpy array)
    """

    if not direction:
        direction = np.eye(3)

    D = np.dtype({'names': [
        'matrix',
        'voxel_size',
        'read_points',
        'read_gap',
        'spokes_hi',
        'spokes_lo',
        'lo_scale',
        'channels',
        'volumes',
        'tr',
        'origin',
        'direction'],
        'formats': [
        ('<i8', (3,)),
        ('<f4', (3,)),
        '<i8',
        '<i8',
        '<i8',
        '<i8',
        '<f4',
        '<i8',
        '<i8',
        '<f4',
        ('<f4', (3,)),
        ('<f4', (9,))]
    })

    info = np.array([(matrix, voxel_size, read_points, read_gap, spokes_hi, spokes_lo, lo_scale,
                      channels, volumes, tr, origin, direction)], dtype=D)

    return info


def nii2h5():
    """
    Converts a nifti file to riesling format .h5 image file

    .. code:: text

        usage: nii2h5 niifile

        nii2h5 converts from nii to h5

        positional arguments:
        input       Input nii image

        optional arguments:
        -h, --help  show this help message and exit
        --out OUT   Output h5 image
    """
    parser = argparse.ArgumentParser(description='nii2h5 converts from nii to h5',
                                     usage='nii2h5 niifile')

    parser.add_argument("input", help="Input nii image")
    parser.add_argument("--out", help="Output h5 image",
                        required=False, type=str)

    args = parser.parse_args()

    print("Opening {}".format(args.input))
    img = sitk.ReadImage(args.input)
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    img_data = sitk.GetArrayFromImage(img)

    info = create_info(matrix=[0, 0, 0],
                       voxel_size=list(spacing),
                       read_points=0, read_gap=0, spokes_hi=0, spokes_lo=0, lo_scale=0,
                       channels=1, volumes=1, origin=list(origin), direction=list(direction))

    output_name = None
    if args.out:
        output_name = args.out
    else:
        output_name = parse_fname(args.input) + '.h5'

    if os.path.isfile(output_name):
        print('{} output already exists'.format(output_name))
        return
    else:
        print('Writing to {}'.format(output_name))
        h5 = h5py.File(output_name, 'w')
        h5.create_dataset('image', data=img_data[np.newaxis, ...])
        h5.create_dataset('info', data=info)
        h5.close()


def h52nii():
    """
    Converts riesling image .h5 file to nifti. 

    .. code:: text

        usage: h52nii h5image

        h52nii converts from h5 to nii

        positional arguments:
        input       Input h5 image

        optional arguments:
        -h, --help  show this help message and exit
        --out OUT   Output image
    """
    parser = argparse.ArgumentParser(description='h52nii converts from h5 to nii',
                                     usage='h52nii h5image')

    parser.add_argument("input", help="Input h5 image")
    parser.add_argument("--out", help="Output image",
                        required=False, type=str)

    args = parser.parse_args()

    print("Opening {}".format(args.input))
    f = h5py.File(args.input, 'r')
    info = f['info'][:]

    data = f['image'][0, ...]
    f.close()

    voxel_size = np.array(info['voxel_size'][0], dtype=float)
    origin = np.array(info['origin'][0], dtype=float)
    direction = np.array(info['direction'][0], dtype=float)

    img = sitk.GetImageFromArray(abs(data))

    img.SetOrigin(origin)
    img.SetSpacing(voxel_size)
    img.SetDirection(direction)

    output_name = None
    if args.out:
        output_name = args.out
    else:
        output_name = os.path.splitext(args.input)[0] + '.nii.gz'

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_name)
    print("Saving output to: {}".format(output_name))
    writer.Execute(img)


def h5viewer():
    """
    Simple static 3-plane viewer of .h5 image data. Will read .h5 files in the riesling format, i.e. with a dataset named ``image``.

    .. code:: text

        usage: h5viewer file.h5

        h5viewer

        positional arguments:
        H5          File input

        optional arguments:
        -v,         Volume to show (default=0)
        -e,         Echo to show (default=0)
        -h, --help  Show this help message and exit
    """
    parser = argparse.ArgumentParser(
        description="h5viewer", usage='h5viewer file.h5')
    parser.add_argument("h5file", metavar="H5", help="File input", type=str)
    parser.add_argument("-v", help="Volume", type=int, default=0)
    parser.add_argument("-e", help="Echo", type=int, default=0)

    args = parser.parse_args()
    I, _ = read_image_h5(args.h5file, args.vol, args.echo)

    print("Displaying %s" % f)
    plot_3plane(abs(I), title=f, cmap='gray', vmin=None, vmax=None)
    plt.tight_layout()
    plt.show()
