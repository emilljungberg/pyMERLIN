from pymerlin.utils import parse_combreg
import numpy as np
import pickle
import warnings

import imageio
from IPython.display import display, HTML
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from matplotlib.backends.backend_agg import FigureCanvasAgg


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


def gif_animation(reg_out, images, out_name='animation.gif', tnav=None, t0=0, max_d=None, max_r=None):
    """
    Creates gif animation of registration results and navigator images.

    Input:
        - reg_out: pickle file with registration output
        - images: List of navigator slices
        - out_name: Output name
        - tnav: Duration of each navigator in seconds for time axis (None)
        - t0: Time offset due to WASPI and dda (0)
        - max_d: Max displacement on y-axis (None)
        - max_r: Max rotation on y-axis (None)
    """

    combreg = pickle.load(open(reg_out, 'rb'))
    num_navigators = len(images)

    if len(combreg) != num_navigators:
        raise TypeError

    all_reg = {'rx': [], 'ry': [], 'rz': [], 'dx': [], 'dy': [], 'dz': []}
    for k in all_reg.keys():
        for i in range(len(combreg)):
            all_reg[k].append(combreg[i][k])

    plt.style.use('default')
    plt.rcParams.update({'font.size': 14})

    # Time axis
    if tnav:
        plot_xlabel = 'Time [s]'
        t = np.arange(num_navigators)*tnav + t0
    else:
        plot_xlabel = 'Navigator'
        t = np.arange(num_navigators)

    # Translations
    if not max_d:
        max_d = np.ceil(np.max([all_reg['dx'], all_reg['dy'], all_reg['dz']]))
    d_axis = [0, max(t), -max_d, max_d]

    if not max_r:
        max_r = np.ceil(np.rad2deg(
            np.max([all_reg['rx'], all_reg['ry'], all_reg['rz']])))
    r_axis = [0, max(t), -max_r, max_r]

    use_raster = True
    raster_order = -10

    def my_plot(img_idx):
        fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        canvas = FigureCanvasAgg(fig)
        spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
        axes = {}

        for (i, ax) in enumerate(['x', 'y', 'z']):
            # Translation
            axes['d%s' % ax] = fig.add_subplot(
                spec[i, 0], rasterized=use_raster)
            plt.plot(t, all_reg['d%s' % ax], linewidth=3, color='C%d' % i)
            if i == 0:
                plt.title('Translation')
            plt.ylabel(r'$\Delta_%s$ [mm]' % ax)
            plt.plot([t[img_idx], t[img_idx]], [-max_d, max_d], '--k')
            plt.gca().set_rasterization_zorder(raster_order)
            if i == 2:
                plt.xlabel(plot_xlabel)
            plt.grid()
            plt.axis(d_axis)

            # Rotation
            axes['r%s' % ax] = fig.add_subplot(
                spec[i, 1], rasterized=use_raster)
            plt.plot(t, np.rad2deg(
                all_reg['r%s' % ax]), linewidth=3, color='C%d' % i)
            if i == 0:
                plt.title('Rotation [deg]')
            plt.ylabel(r'$\alpha_%s$ [deg]' % ax)
            plt.plot([t[img_idx], t[img_idx]], [-max_r, max_r], '--k')
            plt.gca().set_rasterization_zorder(raster_order)
            if i == 2:
                plt.xlabel(plot_xlabel)
            plt.grid()
            plt.axis(r_axis)

        axes['img'] = fig.add_subplot(spec[0:3, 2], rasterized=use_raster)
        plt.imshow(images[img_idx], cmap='gray')
        plt.title('Navigator (%d/%d)' % (img_idx+1, num_navigators))
        plt.axis('off')
        plt.gca().set_rasterization_zorder(raster_order)

        return (fig, canvas)

    # Produce the frames
    gif_frames = []
    max_v = 0
    for i in range(num_navigators):
        print("Processing frame: %d/%d" % (i+1, num_navigators))
        fig, canvas = my_plot(i)
        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf)
        if np.amax(X) > max_v:
            max_v = np.amax(X)
        gif_frames.append(X)
        plt.close(fig)

    # Scale data
    uint8_frames = []
    for i in range(num_navigators):
        uint8_frames.append(np.array(gif_frames[i]/max_v*255, dtype=np.uint8))

    print("Saving output to: {}".format(out_name))
    imageio.mimsave(out_name, uint8_frames)


def report_plot(combreg, maxd, maxr, navtr=None, bw=False):

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
        rot_ax.set_xlabel('Interleave')
        d_ax.set_xlabel('Interleave')

    plt.tight_layout()

#### Old ####


def compare_timeseries(TS, title):
    fig = plt.figure(figsize=(14, 4))
    [nx, ny, nz, nt] = np.shape(TS)

    fig.add_subplot(3, 1, 1)
    imshow3(TS[int(nx/2), :, :, :]-np.repeat(TS[int(nx/2), :, :, 0:1],
                                             repeats=nt, axis=-1), nt, 1, vmin=-.3, vmax=.3)
    plt.title(title, size=20)

    fig.add_subplot(3, 1, 2)
    imshow3(np.rot90(TS[:, int(ny/2), :, :]-np.repeat(TS[:, int(ny/2),
                                                         :, 0:1], repeats=nt, axis=-1)), nt, 1, vmin=-.3, vmax=.3)

    fig.add_subplot(3, 1, 3)
    imshow3(np.rot90(TS[:, :, int(nz/2), :]-np.repeat(TS[:, :, int(nz/2),
                                                         0:1], repeats=nt, axis=-1), 3), nt, 1, vmin=-.3, vmax=.3)

    plt.tight_layout()
    plt.show()


def plot_rotation_translation(df_list, names):
    """
    Plot rotation and translation result from registration outputs

    Input:
        - df_list: Single dictionary or list of dicts
        - names: Single string name or list of strings
    """

    if not isinstance(df_list, list):
        df_list = [df_list]

    if not isinstance(names, list):
        names = [names]

    fig = plt.figure(figsize=(13, 6))
    x = np.arange(np.shape(df_list[0])[1])

    for (i, k) in enumerate(['X', 'Y', 'Z']):
        fig.add_subplot(2, 3, i+1)
        key = 'Versor %s' % k
        for (df, name) in zip(df_list, names):
            plt.plot(x, np.rad2deg(df.loc[key, :]), '-o', label=name)

        if i == 0:
            plt.legend()

        plt.grid()
        plt.xlabel('Interleave')
        plt.ylabel('Angle [deg]')
        plt.title('Rotation in %s' % k, size=16)

    for (i, k) in enumerate(['X', 'Y', 'Z']):
        fig.add_subplot(2, 3, i+1+3)
        key = 'Trans %s' % k
        for (df, name) in zip(df_list, names):
            plt.plot(x, df.loc[key, :], '-o', label=name)

        if i == 0:
            plt.legend()

        plt.grid()
        plt.xlabel('Moving Interleave num')
        plt.ylabel('Distance [mm]')
        plt.title('Translation in %s' % k, size=16)

    plt.tight_layout()
    plt.show()
