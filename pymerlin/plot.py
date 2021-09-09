# -*- coding: utf-8 -*-
"""
Plotting tools for 3D and 4D data.
"""

from builtins import ValueError
import os
import pickle
import warnings

import imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm

from .utils import parse_combreg


def plot_3plane(I, title='', cmap='gray', vmin=None, vmax=None):
    """3 plane plot of 3D image data

    Args:
        I (array): 3D image array
        title (str, optional): Plot title. Defaults to ''.
        cmap (str, optional): colormap. Defaults to 'gray'.
        vmin (int, optional): Lower window limit. Defaults to None.
        vmax (int, optional): Upper window limit. Defaults to None.
    """

    [nx, ny, nz] = np.shape(I)

    fig = plt.figure(figsize=(12, 6), facecolor='black')
    fig.add_subplot(1, 3, 1)
    plt.imshow(I[int(nx/2), :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(' ')

    fig.add_subplot(1, 3, 2)
    plt.imshow(I[:, int(ny/2), :], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(title, color='w', size=20)

    fig.add_subplot(1, 3, 3)
    plt.imshow(I[:, :, int(nz/2)], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title(' ')

    # plt.show()


def timeseries_video(img_ts, interval=100, title=''):
    """Show a time series as a video in a Jupyter Notebook

    To view animation in notebook run
    ```python
    from IPython.display import HTML
    video = timeseries_video(TS)
    HTML(video)
    ```

    Args:
        img_ts (array): Time series sliced to desired view [nx,ny,nt]
        interval (int, optional): Framerate. Defaults to 100.
        title (str, optional): Title. Defaults to ''.

    Returns:
        video: HTML video object
    """

    fig, ax = plt.subplots()
    img = ax.imshow(img_ts[:, :, 0], cmap='gray')
    plt.axis('off')
    plt.title('')
    [nx, ny, nt] = np.shape(img_ts)

    def init():
        img.set_data(img_ts[:, :, 0])
        return (img,)

    def animate(i):
        t = i
        img.set_data(img_ts[:, :, t])
        ax.set_title('%s (t=%d/%d)' % (title, t, nt-1))
        return (img,)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nt, interval=interval, blit=True)
    plt.close()

    return anim.to_html5_video()


def imshow3(I, ncol=None, nrow=None, cmap='gray', vmin=None, vmax=None, order='col'):
    """Tiled plot of 3D data

    Args:
        I (3D array): Data to plot, expanding along last dimentsion
        ncol (int, optional): Number of columns. Defaults to None.
        nrow (int, optional): Number of rows. Defaults to None.
        cmap (str, optional): Matplotlob colormap. Defaults to 'gray'.
        vmin (float, optional): Color range lower limit. Defaults to None.
        vmax (float, optional): Color range higher limit. Defaults to None.
        order (str, optional): Row or column order. Defaults to 'col'.

    Returns:
        np.array: Tiled array
    """

    """Multi-pane plot of 3D data

    Inspired by the matlab function imshow3.
    https://github.com/mikgroup/espirit-matlab-examples/blob/master/imshow3.m

    Expands the 3D data along the last dimension. Data is shown on the current matplotlib axis.

        - ncol: Number of columns
        - nrow: Number of rows
        - cmap: colormap ('gray')

    Output:
        - I3: Same image as shown

    Args:
        I (array): 3D array with 2D images stacked along last dimension
        ncol (int, optional): Number of columns. Defaults to None.
        nrow (int, optional): Number of rows. Defaults to None.
        cmap (str, optional): Colormap. Defaults to 'gray'.
        vmin (innt, optional): Lower window limit. Defaults to None.
        vmax (int, optional): Upper window limit. Defaults to None.
        order (str, optional): Plot order 'col/row'. Defaults to 'col'.

    Returns:
        array: Image expanded along the third dimension
    """

    [nx, ny, n] = np.shape(I)
    if (not nrow) and (not ncol):
        nrow = int(np.floor(np.sqrt(n)))
        ncol = int(n/nrow)
    elif not ncol:
        ncol = int(np.ceil(n/nrow))
    elif not nrow:
        nrow = int(np.ceil(n/ncol))

    I3 = np.zeros((ny*nrow, nx*ncol))

    i = 0
    if order == 'col':
        for ix in range(ncol):
            for iy in range(nrow):
                try:
                    I3[iy*ny:(iy+1)*ny, ix*nx:(ix+1)*nx] = I[:, :, i]
                except:
                    warnings.warn('Warning: Empty slice. Setting to 0 instead')
                    continue

                i += 1

    else:
        for iy in range(nrow):
            for ix in range(ncol):
                try:
                    I3[iy*ny:(iy+1)*ny, ix*nx:(ix+1)*nx] = I[:, :, i]
                except:
                    warnings.warn('Warning: Empty slice. Setting to 0 instead')
                    continue

                i += 1

    plt.imshow(I3, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')

    return I3


def reg_animation(reg_out, images, out_name='animation.gif', slice_pos=None, tnav=None, t0=0, max_d=None, max_r=None, vmax=1, nrot=0):
    """Animation of registration results

    Args:
        reg_out (str): File name to pickle file
        images (np.array): Array of images to display (nx,ny,nz,nt)
        out_name (str, optional): Output suffix .gif or .mp4. Defaults to 'animation.gif'. 
        slice_pos (array, optional): Slice positions. Defaults to center slices.
        tnav (float, optional): Navigator duration to display x-axis with time. Defaults to None.
        t0 (int, optional): Starting time. Defaults to 0.
        max_d (float, optional): Limit of translation plot. Defaults to None.
        max_r (float, optional): Limit of translation plot. Defaults to None.
        vmax (float, optional): Maximum display range (0-1). Defaults to 1

    Raises:
        TypeError: If number of navigator images in `images` is not the same as number of registration objects in pickle file.
    """

    combreg = pickle.load(open(reg_out, 'rb'))
    num_navigators = images.shape[3]

    if len(combreg) != num_navigators:
        raise TypeError

    all_reg = parse_combreg(combreg)

    plt.style.use('default')
    plt.rcParams.update({'font.size': 14})

    # Time axis
    if tnav:
        plot_xlabel = 'Time [s]'
        t = np.arange(num_navigators)*tnav + t0
    else:
        plot_xlabel = 'Navigator'
        t = np.arange(num_navigators)

    if not max_d:
        max_d = np.ceil(
            np.max(np.abs([all_reg['dx'], all_reg['dy'], all_reg['dz']])))
    d_axis = [0, max(t), -max_d, max_d]

    if not max_r:
        max_r = np.ceil(np.rad2deg(
            np.max(np.abs([all_reg['rx'], all_reg['ry'], all_reg['rz']]))))
    r_axis = [0, max(t), -max_r, max_r]

    if slice_pos is None:
        nx, ny, nz, nt = images.shape
        slice_pos = [int(nx/2), int(ny/2), int(nz/2)]

    use_raster = True
    raster_order = -10

    max_img = np.max(abs(images[:]))

    def plot_frame(img_idx):
        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        canvas = FigureCanvasAgg(fig)
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
        axes = {}

        d_ax = fig.add_subplot(spec[0, 0], rasterized=use_raster)
        r_ax = fig.add_subplot(spec[0, 1], rasterized=use_raster)

        for (i, ax) in enumerate(['x', 'y', 'z']):
            d_ax.plot(t, all_reg['d%s' % ax], linewidth=3, color='C%d' % i)
            d_ax.plot([t[img_idx], t[img_idx]], [-max_d, max_d], '--k')
            plt.gca().set_rasterization_zorder(raster_order)

            r_ax.plot(t, np.rad2deg(
                all_reg['r%s' % ax]), linewidth=3, color='C%d' % i)
            r_ax.plot([t[img_idx], t[img_idx]], [-max_r, max_r], '--k')
            plt.gca().set_rasterization_zorder(raster_order)

        d_ax.set_title('Translation')
        d_ax.set_ylabel(r'$\Delta_%s$ [mm]' % ax)
        d_ax.set_xlabel(plot_xlabel)
        d_ax.axis(d_axis)
        d_ax.grid()

        r_ax.set_title('Rotation')
        r_ax.set_ylabel(r'$\alpha_%s$ [deg]' % ax)
        r_ax.set_xlabel(plot_xlabel)
        r_ax.axis(r_axis)
        r_ax.grid()

        axes['img'] = fig.add_subplot(spec[1:3, 0:2], rasterized=use_raster)
        img_view = np.concatenate([
            np.rot90(images[slice_pos[0], :, :, img_idx], nrot),
            np.rot90(images[:, slice_pos[1], :, img_idx], nrot),
            np.rot90(images[:, :, slice_pos[2], img_idx], nrot)], axis=1)
        plt.imshow(img_view/max_img*255, cmap='gray', vmin=0, vmax=vmax*255)
        plt.title('Navigator (%d/%d)' % (img_idx+1, num_navigators))
        plt.axis('off')
        plt.gca().set_rasterization_zorder(raster_order)

        return (fig, canvas)

    # Determine output type
    suffix = os.path.splitext(os.path.basename(out_name))[-1]
    out_type = None
    if suffix == '.gif':
        out_type = 'gif'
    elif suffix == '.mp4':
        out_type = 'mp4'
    else:
        raise ValueError("Output name must be .gif or .mp4")

    # Produce the frames
    frames = []

    if out_type == 'mp4':
        writer = imageio.get_writer(
            out_name, format='FFMPEG', mode='I', fps=10)

    print("Processing frames")
    for i in tqdm(range(num_navigators)):
        fig, canvas = plot_frame(i)
        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf, dtype=np.uint8)
        if out_type == 'mp4':
            writer.append_data(X)
        else:
            frames.append(X)
        plt.close(fig)

    # Scale data
    # uint8_frames = []
    # for i in range(num_navigators):
        # uint8_frames.append(np.array(frames[i], dtype=np.uint8))

    print("Saving output to: {}".format(out_name))
    if out_type == 'mp4':
        writer.close()
    else:
        imageio.mimsave(out_name, frames)


def report_plot(combreg, maxd, maxr, navtr=None, bw=False):
    """Plot registration results

    Args:
        combreg (str): Filename of registration results
        maxd (float): Max displacement for y-limit
        maxr (float): Max rotation for y-limit
        navtr (float, optional): Duration of navigator for time axis. Defaults to None.
        bw (bool, optional): Plot in black and white. Defaults to False.
    """

    # Summarise statistics
    all_reg = parse_combreg(combreg)

    if bw:
        plt.style.use('grayscale')
    else:
        plt.style.use('default')

    fig = plt.figure(figsize=(12, 4), facecolor='w')
    plt.rcParams.update({'font.size': 16})

    # Axis limits
    max_d = float(maxd)
    max_r = float(maxr)
    if not max_d:
        max_d = np.ceil(
            np.max(np.abs([all_reg['dx'], all_reg['dy'], all_reg['dz']])))
    if not max_r:
        max_r = np.ceil(np.rad2deg(
            np.max(np.abs([all_reg['rx'], all_reg['ry'], all_reg['rz']]))))

    x = list(range(len(combreg)))
    if navtr:
        x *= navtr

    d_ax = fig.add_subplot(1, 2, 1)
    rot_ax = fig.add_subplot(1, 2, 2)

    for (i, ax) in enumerate(['x', 'y', 'z']):
        d_ax.plot(x, all_reg['d%s' % ax],
                  linewidth=3, label=ax,)

        rot_ax.plot(x, np.rad2deg(all_reg['r%s' % ax]),
                    linewidth=3, label=ax)

    d_ax.axis([0, max(x), -max_d, max_d])
    d_ax.set_ylabel(r'$\Delta$ [mm]')
    d_ax.grid()
    d_ax.set_title('Translation')
    d_ax.legend()

    rot_ax.axis([0, max(x), -max_r, max_r])
    rot_ax.grid()
    rot_ax.set_ylabel(r'$\alpha$ [deg]')
    rot_ax.set_title('Rotation')
    rot_ax.legend()

    if navtr:
        rot_ax.set_xlabel('Time [s]')
        d_ax.set_xlabel('Time [s]')
    else:
        rot_ax.set_xlabel('Navigator')
        d_ax.set_xlabel('Navigator')

    plt.tight_layout()
