#!/usr/bin/env python3

"""
Main script to run MERLIN functions on the command line. Uses ``.h5`` files as input, assuming that the dataset ``image`` is occupied by the image data. 

The executable works like the git command with subcommands.

.. code:: bash

    usage: pymerlin <command> [<args>]

        Available commands are:
            reg         Register data
            merge       Merge registration into series
            moco        Run moco
            report      View report of data
            metric      Image metric analysis
            view        View h5 file
            gif         Navigator and registration animation
            ssim        Calculate Structural Similarity Index Measure
            aes         Calculate Average Edge Strength
            nrmse       Calculate Normalised Root Mean Squared Error
            tukey       Applies Tukey filter to radial k-space data

To get more help for a specific command add ``-h``.

.. code:: bash

    >> pymerlin reg -h

    usage: pymerlin reg [<args>]

    MERLIN Registration

    optional arguments:
    -h, --help            show this help message and exit
    --fixed FIXED         Fixed image
    --moving MOVING       Moving image
    --reg REG             Registration parameters
    --log LOG             Registration history log
    --fixout FIXOUT       Name of fixed image output
    --moveout MOVEOUT     Name of registered moving image output
    --rad RAD             Radius of fixed mask
    --thr THR             Low image threshold
    --sigma SIGMA [SIGMA ...]
                            List of sigmas
    --shrink SHRINK [SHRINK ...]
                            Shrink factors
    --metric METRIC       Image metric
    --verbose VERBOSE     Log level (0,1,2)

"""

import argparse
import logging
import os
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from .dataIO import (arg_check_h5, arg_check_nii, make_3D, parse_fname,
                     read_image_h5)
from .iq import aes, nrmse, ssim, gradient_entropy
from .moco import moco_combined, moco_single, moco_sw
from .plot import gif_animation, report_plot
from .reg import ants_pyramid, histogram_threshold_estimator
from .utils import make_tukey


class PyMerlin_parser(object):
    """
    Class to produce a subcommand arg parser like git for pyMERLIN

    Inspired by: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description='MERLIN Python tools',
                                         usage='''pymerlin <command> [<args>]

    Available commands are:
        reg         Register data
        merge       Merge registration into series
        moco        Run moco
        report      View report of data
        metric      Image metric analysis
        view        View h5 file
        gif         Navigator and registration animation
        ssim        Calculate Structural Similarity Index Measure
        aes         Calculate Average Edge Strength
        nrmse       Calculate Normalised Root Mean Squared Error
        tukey       Applies Tukey filter to radial k-space data
    '''
                                         )

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        # Here we check if out object (the class) has a function with the given name
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # Call the method
        getattr(self, args.command)()

    def reg(self):
        parser = argparse.ArgumentParser(description='MERLIN Registration',
                                         usage='pymerlin reg [<args>]')

        parser.add_argument("--fixed", help="Fixed image",
                            required=True, type=arg_check_h5)
        parser.add_argument("--moving", help="Moving image",
                            required=True, type=arg_check_h5)
        parser.add_argument(
            "--reg", help="Registration parameters", required=False, type=str, default=None)
        parser.add_argument("--log", help="Registration history log",
                            required=False, type=str, default=None)
        parser.add_argument(
            "--fixout", help="Name of fixed image output", required=False, type=arg_check_nii)
        parser.add_argument(
            "--moveout", help="Name of registered moving image output", required=False, type=arg_check_nii)
        parser.add_argument("--rad", help="Radius of fixed mask",
                            required=False, default=1.0, type=float)
        parser.add_argument("--thr", help="Low image threshold",
                            required=False, default=0, type=float)
        parser.add_argument("--sigma", help="List of sigmas",
                            required=False, default=[2, 1, 0], nargs="+", type=int)
        parser.add_argument("--shrink", help="Shrink factors",
                            required=False, default=[4, 2, 1], nargs="+", type=int)
        parser.add_argument("--metric", help="Image metric",
                            required=False, default="MI")
        parser.add_argument(
            "--verbose", help="Log level (0,1,2)", default=2, type=int)

        # Since we are inside the subcommand now we skip the first two
        # arguments on the command line
        args = parser.parse_args(sys.argv[2:])
        main_reg(args)

    def merge(self):
        parser = argparse.ArgumentParser(description='Append to registration or initialise series',
                                         usage='pymerlin merge [<args>]')
        parser.add_argument("--input", nargs='+', help="Reg input. Initialize by setting input to 0",
                            required=False)
        parser.add_argument("--reg", help="Output reg object to save or append to",
                            required=True)
        parser.add_argument("--spi", help="Spokes per interleave",
                            type=int, required=True)
        parser.add_argument(
            "--verbose", help="Log level (0,1,2)", default=2, type=int)

        args = parser.parse_args(sys.argv[2:])
        main_merge(args)

    def moco(self):
        parser = argparse.ArgumentParser(
            description="Moco of complete series interleave", usage='pymerlin moco [<args>]')
        parser.add_argument("--input", help="Input H5 k-space",
                            required=True, type=arg_check_h5)
        parser.add_argument("--output", help="Output correct H5 k-space",
                            required=True, type=arg_check_h5)
        parser.add_argument(
            "--reg", help="All registration parameters in combined file", required=True, type=str, default=None)
        parser.add_argument(
            "--nseg", help="Segments per interleave for sliding window", required=False, type=int, default=None
        )
        parser.add_argument(
            "--verbose", help="Log level (0,1,2)", default=2, type=int)

        args = parser.parse_args(sys.argv[2:])
        main_moco(args)

    def thr(self):
        parser = argparse.ArgumentParser(
            description="Quick navigator background threshold estimator", usage='pymerlin thr [<args>]')
        parser.add_argument("--input", help="Input H5 navigator",
                            required=True, type=arg_check_h5)
        parser.add_argument("--nbins", help="Number of bins",
                            type=int, default=200)
        parser.add_argument("--plot", help="Show plot", action='store_true')

        args = parser.parse_args(sys.argv[2:])
        main_thr(args)

    def report(self):
        parser = argparse.ArgumentParser(
            description="Moco report", usage='pymerlin report [<args>]')
        parser.add_argument("--reg", help="Combined registration object",
                            required=True)
        parser.add_argument(
            "--out", help="Output name of figure (.png)", required=False, type=str, default='regstats.png')
        parser.add_argument(
            "--navtr", help="Navigator duration (s)", required=False, type=float)
        parser.add_argument(
            "--maxd", help="Max y-range translation", required=False, default=0)
        parser.add_argument(
            "--maxr", help="Max y-range rotation", required=False, default=0)
        parser.add_argument("--bw", action='store_true', default=False,
                            required=False, help="Plot in black and white")
        args = parser.parse_args(sys.argv[2:])
        main_report(args)

    def metric(self):
        parser = argparse.ArgumentParser(
            description="Image metrics", usage="pymerlin metric [<args>]")
        parser.add_argument("--input", help="Image input", required=True)

        args = parser.parse_args(sys.argv[2:])
        main_metric(args)

    def view(self):
        parser = argparse.ArgumentParser(
            description="View h5 file", usage='pymerlin view H5FILE')
        parser.add_argument("input", help="H5 input file",
                            type=str, required=True)

        args = parser.parse_args(sys.argv[2:])
        main_view(args)

    def gif(self):
        parser = argparse.ArgumentParser(
            description="Make animation from navigator and reg results", usage='pymerlin gif [<args>]')
        parser.add_argument("--reg", help="Combined registration object",
                            required=True)
        parser.add_argument("--nav", help="Navigator folder", required=True)
        parser.add_argument("--out", help="Output gif name",
                            required=False, default="reg_animation.gif")
        parser.add_argument("--axis", help="Slice axis (x,y,z)",
                            required=False, default='z')
        parser.add_argument(
            "--slice", help="Slice to plot (def middle)", required=False, default=None)
        parser.add_argument("--rot", help="Rotations to slice",
                            required=False, default=0, type=int)
        parser.add_argument(
            "--navtr", help="Navigator duration (s)", required=False, type=float)
        parser.add_argument("--t0", help="Time offset",
                            required=False, default=0)
        parser.add_argument(
            "--maxd", help="Max y-range translation", required=False, default=None, type=float)
        parser.add_argument(
            "--maxr", help="Max y-range rotation", required=False, default=None, type=float)

        args = parser.parse_args(sys.argv[2:])
        main_gif(args)

    def ssim(self):
        parser = argparse.ArgumentParser(
            description="Calculate Structural Similarity Index Measure (SSIM)", usage="pymerlin ssim [<args>]")
        parser.add_argument('img1', type=str,
                            help='Reference image')
        parser.add_argument('img2', type=str,
                            help='Comparison image')
        parser.add_argument('--kw', required=False, default=11,
                            type=int, help='Kernel width')
        parser.add_argument('--sigma', required=False, default=0.0,
                            type=float, help='Sigma for Gaussian kernel')
        parser.add_argument('--mask', required=False,
                            default=None, help='Brain mask')
        parser.add_argument('--out', required=False,
                            default='ssim.nii.gz', type=str, help='Output filename')

        args = parser.parse_args(sys.argv[2:])
        main_ssim(args)

    def aes(self):
        parser = argparse.ArgumentParser(
            description="Calculate the Average Edge Strength (AES)", usage="pymerlin aes [<args>]")
        parser.add_argument('img', type=str, help='Input image')
        parser.add_argument('--mask', type=str,
                            help='Brain mask', required=False, default=None)
        parser.add_argument('--canny', type=str,
                            help='Canny edge mask', required=False, default=None)
        parser.add_argument(
            '--sigma', type=float, help='Canny edge filter sigma', required=False, default=2)
        parser.add_argument('--out', type=str,
                            help='Output folder', required=False, default='.')

        args = parser.parse_args(sys.argv[2:])
        main_aes(args)

    def nrmse(self):
        parser = argparse.ArgumentParser(
            description="Calculate the Normalised Root Mean Squared Error (NRMSE)", usage="pymerlin nrmse [<args>]")
        parser.add_argument("--ref", type=str,
                            help="Reference image", required=True)
        parser.add_argument("--comp", type=str,
                            help="Comparison image", required=True)
        parser.add_argument("--mask", type=str, help="Mask", required=True)
        parser.add_argument("--out", type=str,
                            help="Output folder", required=False, default='.')

        args = parser.parse_args(sys.argv[2:])
        main_nrmse(args)

    def tukey(self):
        parser = argparse.ArgumentParser(
            description="Applies a tukey filter to radial k-space data",
            usage="pymerlin tukey [<args>]")
        parser.add_argument("--input", type=str,
                            help="Input data", required=True)
        parser.add_argument("--output", type=str,
                            help="Output data", required=True)
        parser.add_argument("--alpha", required=False,
                            type=float, default=0.5, help="Filter width")

        args = parser.parse_args(sys.argv[2:])
        main_tukey(args)

    def get_args(self):
        return self.outargs


def main_reg(args):
    if not args.reg:
        fix_base = os.path.splitext(os.path.basename(args.fixed))
        move_base = os.path.splitext(os.path.basename(args.moving))
        reg_name = "{}_2_{}_reg.p".format(move_base, fix_base)
    else:
        reg_name = args.reg

    r, rout, reg_fname = ants_pyramid(fixed_image_fname=args.fixed,
                                      moving_image_fname=args.moving,
                                      moco_output_name=args.moveout,
                                      fixed_output_name=args.fixout,
                                      reg_par_name=reg_name,
                                      iteration_log_fname=args.log,
                                      fixed_mask_radius=args.rad,
                                      threshold=args.thr,
                                      winsorize=None,
                                      sigmas=args.sigma,
                                      shrink=args.shrink,
                                      metric=args.metric,
                                      verbose=args.verbose)


def main_thr(args):
    img, spacing = read_image_h5(args.input)
    thr = histogram_threshold_estimator(img, args.plot, args.nbins)
    print(thr)


def main_report(args):
    combreg = pickle.load(open(args.reg, 'rb'))

    report_plot(combreg, args.maxd, args.maxr, args.navtr, args.bw)

    # Check filename
    out_name = args.out
    fname, ext = os.path.splitext(out_name)
    if ext != '.png':
        print("Warning: output extension is not .png")
        out_name = fname + '.png'
        print("Setting output name to: {}".format(out_name))

    plt.savefig(out_name, dpi=300)
    plt.show()


def main_moco(args):
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=log_level[args.verbose], datefmt="%I:%M:%S")

    reg_list = pickle.load(open(args.reg, 'rb'))

    if isinstance(reg_list, dict):
        moco_single(args.input, args.output, reg_list)
    elif args.nseg:
        moco_sw(args.input, args.output, reg_list, args.nseg)
    else:
        moco_combined(args.input, args.output, reg_list)


def main_merge(args):
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=log_level[args.verbose], datefmt="%I:%M:%S")

    if not args.input:
        logging.info("Initializing new reg structure")

        if os.path.exists(args.reg):
            logging.warning("%s already exists" % args.reg)
            raise Exception("Cannot overwrite file")

        D = {'R': np.eye(3),
             'rx': 0,
             'ry': 0,
             'rz': 0,
             'dx': 0,
             'dy': 0,
             'dz': 0,
             'spi': args.spi}

        L = []
        L.append(D)
        pickle.dump(L, open(args.reg, 'wb'))
        logging.info("New reg structure saved to {}".format(args.reg))

    else:
        logging.info("Opening {}".format(args.reg))
        dlist = pickle.load(open(args.reg, 'rb'))

        for input in args.input:
            logging.info("Opening {}".format(input))
            in_dict = pickle.load(open(input, 'rb'))
            in_dict['spi'] = args.spi
            dlist.append(in_dict)

            nreg = len(dlist)

            logging.info("Adding as reg object number {}".format(nreg))

        logging.info("Writing combined reg object back to {}".format(args.reg))
        pickle.dump(dlist, open(args.reg, 'wb'))


def main_view(args):
    plt.style.use('dark_background')

    f = args.input
    h5 = h5py.File(f, 'r')
    fname = os.path.basename(f)
    img = h5['data/0000']

    fig = plt.figure(figsize=(12, 6), facecolor='black')
    nx, ny, nz = np.shape(img)
    fig.add_subplot(1, 3, 1)
    plt.imshow(abs(img[int(nx/2), :, :]), cmap='gray')
    plt.axis('off')
    fig.add_subplot(1, 3, 2)
    plt.imshow(abs(img[:, int(ny/2), :]), cmap='gray')
    plt.axis('off')
    plt.title(fname, size=20)
    fig.add_subplot(1, 3, 3)
    plt.imshow(abs(img[:, :, int(nz/2)]), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main_metric(args):
    img = nib.load(args.input).get_fdata()
    GE = gradient_entropy(img)

    print(GE)


def main_gif(args):

    nav_dir = args.nav
    files = os.listdir(nav_dir)
    fbase = ''
    string_match = '-nav0-'
    for f in files:
        if string_match in f:
            fbase = f.split(string_match)

    file_tmpl = os.path.join(nav_dir, fbase[0] + '-nav%d-' + fbase[1])
    num_files = len(files)

    ax = args.axis
    nrot = int(args.rot)
    slice_idx = args.slice
    slice_ax = {'x': 0, 'y': 1, 'z': 2}

    images = []
    for i in range(num_files):
        img, spacing = read_image_h5(file_tmpl % i, vol=0)

        if not slice_idx:
            img_size = img.shape
            slice_idx = img_size[slice_ax[ax]]/2

        images.append(
            np.rot90(abs(np.take(img, int(slice_idx), axis=slice_ax[ax])), nrot))

    # Check filename
    out_name = args.out
    fname, ext = os.path.splitext(out_name)
    if ext != '.gif':
        print("Warning: output extension is not .gif")
        out_name = fname + '.gif'
        print("Setting output name to: {}".format(out_name))

    gif_animation(args.reg, images, out_name=out_name,
                  tnav=args.navtr, t0=0, max_d=args.maxd, max_r=args.maxr)


def main_ssim(args):
    nii1 = nib.load(args.img1)
    image1 = nii1.get_fdata()
    image2 = nib.load(args.img2).get_fdata()

    if len(image1.shape) > 3:
        image1 = image1[..., 0]
    if len(image2.shape) > 3:
        image2 = image2[..., 0]

    if args.mask:
        mask = nib.load(args.mask).get_fdata()
        image1 *= mask
        image2 *= mask

    mssim, S = ssim(image1, image2, kw=args.kw, sigma=args.sigma)

    if args.mask:
        S *= mask

    mssim = np.mean(S[mask == 1])

    ssim_nii = nib.Nifti1Image(S, nii1.affine)
    nib.save(ssim_nii, args.out)
    print('MSSIM: {}'.format(mssim))
    print('Saving SSIM to: {}'.format(args.out))


def main_aes(args):

    nii = nib.load(args.img)

    img = make_3D(nib.load(args.img).get_fdata())

    if args.mask:
        mask = make_3D(nib.load(args.mask).get_fdata())
    else:
        mask = np.ones_like(img)

    if args.canny:
        canny = make_3D(nib.load(args.canny).get_fdata())
    else:
        canny = None

    img_aes, img_edges, canny_edges = aes(
        img, mask=mask, canny_edges=canny, canny_sigma=args.sigma)

    bname = parse_fname(args.img)

    edges_nii = nib.Nifti1Image(img_edges, nii.affine)
    nib.save(edges_nii, "{}/{}_edges.nii.gz".format(args.out, bname))

    if not args.canny:
        canny_nii = nib.Nifti1Image(canny_edges, nii.affine)
        nib.save(canny_nii, "{}/{}_canny.nii.gz".format(args.out, bname))

    print("Average edge strength")
    print("Image: {}".format(args.img))
    if args.canny:
        print("Canny edges: {}".format(args.canny))
    else:
        print("Canny edges calculated from image")

    print("AES:{}".format(img_aes))


def main_nrmse(args):

    ref_nii = nib.load(args.ref)

    ref_img = make_3D(ref_nii.get_fdata())
    comp_img = make_3D(nib.load(args.comp).get_fdata())
    mask = make_3D(nib.load(args.mask).get_fdata())

    # Normalise images to avoid difference in global scaling
    ref_img /= np.quantile(ref_img[mask == 1], 0.99)
    comp_img /= np.quantile(comp_img[mask == 1], 0.99)

    img_nrmse = nrmse(ref_img, comp_img, mask)

    print("Normalised Root Mean Squared Error (NRMSE)")
    print("Ref Image: {}".format(args.ref))
    print("Comparison Image: {}".format(args.comp))
    print("Mask: {}".format(args.mask))
    print("NRMSE: {}".format(img_nrmse))

    diff_img = ref_img - comp_img

    diff_nii = nib.Nifti1Image(diff_img, ref_nii.affine)

    nib.save(diff_nii, "{}/{}_diff.nii.gz".format(args.out,
             parse_fname(args.comp)))


def main_tukey(args):
    print("Applying tukey filter to {}".format(args.input))
    h5 = h5py.File(args.input, 'r')
    ks = h5['noncartesian'][0, ...]
    info = h5['info'][:]
    traj = h5['trajectory'][:]
    h5.close()

    nspokes, npoints, ndim = traj.shape
    filt = make_tukey(nspokes, a=args.alpha)
    ks_filt = np.transpose(np.transpose(ks, [1, 2, 0])*filt, [2, 0, 1])

    f_dest = h5py.File(args.output, 'w')
    f_dest.create_dataset("info", data=info)

    f_dest.create_dataset("trajectory", data=traj,
                          chunks=np.shape(traj), compression='gzip')

    f_dest.create_dataset("noncartesian", dtype='c8', data=ks_filt[np.newaxis, ...],
                          chunks=np.shape(ks_filt[np.newaxis, ...]), compression='gzip')

    f_dest.close()
    print("Saved data to {}".format(args.output))


def main():
    # Everything is executed by initialising this class.
    # The command line arguments will be parsed and the appropriate function will be called
    PyMerlin_parser()


if __name__ == '__main__':
    main()
