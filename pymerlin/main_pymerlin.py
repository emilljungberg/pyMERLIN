#!/usr/bin/env python3

"""
Main script to run MERLIN functions on the command line. The executable works like the git command with
subcommands. Example:

    main_pymerlin.py reg <args>

By: Emil Ljungberg, KCL, 2020
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

from .dataIO import arg_check_h5, arg_check_nii, read_image_h5
from .plot import gif_animation
from .moco import moco_combined
from .reg import ants_pyramid, histogram_threshold_estimator
from .utils import gradient_entropy


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
        view        View h5 file
        thr         Background threshold estimation
        metric      Image metric analysis
        gif         Navigator and registration animation
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
        parser.add_argument("--input", help="Reg input. Initialize by setting input to 0", type=str,
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
            "--maxd", help="Max y-range translation", required=False, default=None)
        parser.add_argument(
            "--maxr", help="Max y-range rotation", required=False, default=None)
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

    # Look at stats
    all_reg = {'rx': [], 'ry': [], 'rz': [], 'dx': [], 'dy': [], 'dz': []}
    for k in all_reg.keys():
        for i in range(len(combreg)):
            all_reg[k].append(combreg[i][k])

    plt.style.use('default')
    fig = plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 16})

    # Translations
    max_d = float(args.maxd)
    max_r = float(args.maxr)
    if not max_d:
        max_d = np.ceil(np.max([all_reg['dx'], all_reg['dy'], all_reg['dz']]))
    if not max_r:
        max_r = np.ceil(np.rad2deg(
            np.max([all_reg['rx'], all_reg['ry'], all_reg['rz']])))

    x = list(range(len(combreg)))
    if args.navtr:
        x *= args.navtr

    d_axis = [0, max(x), -max_d, max_d]
    r_axis = [0, max(x), -max_r, max_r]

    for (i, ax) in enumerate(['x', 'y', 'z']):
        fig.add_subplot(3, 2, i*2+1)
        plt.plot(x, all_reg['d%s' % ax], linewidth=3, color='C%d' % i)
        plt.axis(d_axis)
        plt.grid()
        plt.ylabel(r'$\Delta_%s$ [mm]' % ax)

        if i == 0:
            plt.title('Translation')
        if i == 2:
            if args.navtr:
                plt.xlabel('Time [s]')
            else:
                plt.xlabel('Interleave')

        fig.add_subplot(3, 2, i*2+2)
        plt.plot(x, np.rad2deg(all_reg['r%s' % ax]),
                 linewidth=3, color='C%d' % i)
        plt.axis(r_axis)
        plt.grid()
        plt.ylabel(r'$\alpha_%s$ [deg]' % ax)

        if i == 0:
            plt.title('Rotation')
        if i == 2:
            if args.navtr:
                plt.xlabel('Time [s]')
            else:
                plt.xlabel('Interleave')

    # Check filename
    out_name = args.out
    fname, ext = os.path.splitext(out_name)
    if ext != '.png':
        print("Warning: output extension is not .png")
        out_name = fname + '.png'
        print("Setting output name to: {}".format(out_name))

    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.show()


def main_moco(args):
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=log_level[args.verbose], datefmt="%I:%M:%S")

    reg_list = pickle.load(open(args.reg, 'rb'))
    moco_combined(args.input, args.output, reg_list)


def main_merge(args):
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=log_level[args.verbose], datefmt="%I:%M:%S")

    if args.input == '0':
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
        logging.info("Opening {}".format(args.input))
        in_dict = pickle.load(open(args.input, 'rb'))
        in_dict['spi'] = args.spi

        logging.info("Opening {}".format(args.reg))
        dlist = pickle.load(open(args.reg, 'rb'))
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

    print("Calculating image metrics for %s" % args.input)
    print("Gradient Entropy: {}".format(GE))


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
        nav0 = h5py.File(file_tmpl % i)
        img = nav0['data']['0000'][:]
        nav0.close()

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


def main():
    # Everything is executed by initialising this class.
    # The command line arguments will be parsed and the appropriate function will be called
    PyMerlin_parser()


if __name__ == '__main__':
    main()
