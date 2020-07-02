import matplotlib.pyplot as plt
from matplotlib import animation, rc
import warnings
import numpy as np
from IPython.display import display, HTML


def plot_3plane(I, title='', cmap='gray', vmin=None, vmax=None):
    """
    Simple 3-plane plotting of 3D data.

    Input:
        - I: 3D numpy array
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

    plt.show()


def timeseries_video(img_ts, interval=100, title=''):
    """
    Show a time series as a video in a Jupyter Notebook

    Input:
        - img_ts: Time series sliced to desired view [nx,ny,nt]

    Output:
        - video: HTML5 video object 

    To view animation in notebook run

        from IPython.display import HTML
        video = timeseries_video(TS)
        HTML(video)

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
    """
    Inspired by the matlab function imshow3.
    https://github.com/mikgroup/espirit-matlab-examples/blob/master/imshow3.m

    Expands the 3D data along the last dimension.

    Data is shown on the current axis.

    Inputs:
        - I: 3D array with 2D images stacked along last dimension
        - ncol: Number of columns
        - nrow: Number of rows
        - cmap: colormap ('gray')

    Output:
        - I3: Same image as shown
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

    plt.imshow(I3, cmap=cmap)
    plt.axis('off')

    return I3


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
