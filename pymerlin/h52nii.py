#!/usr/bin/env python3
import argparse
import os

import h5py
import numpy as np
import SimpleITK as sitk


def main():
    parser = argparse.ArgumentParser(description='MERLIN Registration',
                                     usage='pymerlin reg [<args>]')

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


if __name__ == '__main__':
    main()
