import numpy as np
import sigpy
from sigpy.mri.app import TotalVariationRecon
import tqdm

PHI_GOLD = np.pi*(3-np.sqrt(5))
fibonacciNum = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
                233, 377, 610, 987, 1597, 2584, 4181, 6765]


def piccini_phyllotaxis(n, nint):
    """
    Generate a spiral phyllotaxis trajectory with square root z-modulation
    according to formulation by Piccini et al. Note, this does not give a uniform
    FOV but slightly higher sampling in the x/y plane than along z.

    Args:
        n (int): Number of spokes
        nint (int): Number of interleaves

    Returns:
        [array]: Trajectory

    References:
        Piccini D, et al., Magn Reson Med. 2011;66(4):1049–56.
    """

    # Check inputs
    if n % 2:
        raise ValueError('Number of spokes must be even')

    if nint not in fibbonaciNum:
        raise ValueError('Number of interleaves has to be a Fibonacci number')

    if n % nint:
        raise ValueError('Spokes per interleave must be an integer number')

    spokes_per_int = round(n/nint)
    traj_tmp = np.zeros((n, 3))
    traj = np.zeros((n, 3))

    # Calculate trajectory
    for i in range(n):
        if (i < n/2):
            theta_n = np.pi/2 * np.sqrt(1.0*i/(n/2.0))
            gz_sign = 1
        else:
            theta_n = np.pi/2 * np.sqrt((n-i)/(n/2.0))
            gz_sign = -1

        phi_n = i * PHI_GOLD
        traj_tmp[i, 0] = np.sin(theta_n) * np.cos(phi_n)
        traj_tmp[i, 1] = np.sin(theta_n) * np.sin(phi_n)
        traj_tmp[i, 2] = gz_sign * np.cos(theta_n)

    # Stack the interleaves after eachother
    for i in range(nint):
        if i % 2 == 0:
            for j in range(spokes_per_int):
                idx1 = i * spokes_per_int + j
                idx2 = i + j * nint
                traj[idx1, :] = traj_tmp[idx2, :]

        else:
            for j in range(spokes_per_int):
                idx1 = i*spokes_per_int + j
                idx2 = (n - (nint - i)) - j * nint
                traj[idx1, :] = traj_tmp[idx2, :]

    return traj


def swinbank_phyllotaxis(n, nint):
    """
    Generate a spiral phyllotaxis trajectory with cosine z-modulation
    for uniform spherical sampling.

    Args:
        n (int): Number of spokes
        nint (int): Number of interleaves

    Returns:
        [array]: Trajectory

    References: 
        Swinbank R, Purser RJ., Q J R Meteorol Soc. 2006;132(619):1769–93.
    """

    # Check inputs
    if n % 2:
        raise ValueError('Number of spokes must be even')

    if nint not in fibbonaciNum:
        raise ValueError('Number of interleaves has to be a Fibonacci number')

    if n % nint:
        raise ValueError('Spokes per interleave must be an integer number')

    spokes_per_int = round(n/nint)
    traj_tmp = np.zeros((3, n))
    traj = np.zeros((3, n))

    # Calculate trajectory
    for i in range(n):

        theta_n = np.acos((n/2 - i)/(n/2))
        phi_n = i * PHI_GOLD
        traj_tmp[i, 0] = np.sin(theta_n) * np.cos(phi_n)
        traj_tmp[i, 1] = np.sin(theta_n) * np.sin(phi_n)
        traj_tmp[i, 2] = np.cos(theta_n)

    # Stack the interleaves after eachother
    for i in range(nint):
        if i % 2 == 0:
            for j in range(spokes_per_int):
                idx1 = i * spokes_per_int + j
                idx2 = i + j * nint
                traj[:, idx1] = traj_tmp[:, idx2]
        else:
            for j in range(spokes_per_int):
                idx1 = i*spokes_per_int + j
                idx2 = (n - (nint - i)) - j * nint
                traj[:, idx1] = traj_tmp[:, idx2]


def linear_phyllotaxis(n, nint, sf):
    """Isotropic Phyllotaxis trajectory with linear interleave ordering
    and arbitrary smoothness factor

    Args:
        n (int): Number of spokes
        nint (int): Number of interleaves
        sf (int): Smoothness factor

    Returns:
        array: Trajectory

    References:
        1. Swinbank R, Purser RJ., Q J R Meteorol Soc. 2006;132(619):1769–93.
        2. Piccini D, et al., Magn Reson Med. 2011;66(4):1049–56.
    """

    traj = np.zeros((n, 3))
    spi = int(n/nint)

    i = np.arange(spi)
    phi0 = i * PHI_GOLD * fibonacciNum[sf]
    z0 = 1 - 2*nint*i/(n-1)
    r = 1

    for k in range(nint):
        z = z0 - k*2/(n-1)
        phi = phi0 + k * PHI_GOLD

        theta = np.arccos(z)
        traj[k*spi: (k+1)*spi, 0] = r * np.sin(theta) * np.cos(phi)
        traj[k*spi: (k+1)*spi, 1] = r * np.sin(theta) * np.sin(phi)
        traj[k*spi: (k+1)*spi, 2] = r * z

    return traj


def wong_roos_trajectory(n):
    """3D Radial trajectory as formulated by Wong and Roos

    Args:
        n (int): Number of spokes

    Returns:
        array: Trajectory

    References:
        S. T. S. Wong and M. S. Roos, “A strategy for sampling on a sphere applied to 3D selective RF pulse design,” 
        Magn. Reson. Med., vol. 32, no. 6, pp. 778–784, 1994.
    """

    traj = np.zeros((n, 3))
    ni = np.arange(1, n+1)

    traj[:, 2] = (2*ni - n - 1)/n
    traj[:, 0] = np.cos(np.sqrt(n*np.pi)*np.arcsin(traj[:, 2])
                        )*np.sqrt(1-traj[:, 2]**2)
    traj[:, 1] = np.sin(np.sqrt(n*np.pi)*np.arcsin(traj[:, 2])
                        ) * np.sqrt(1-traj[:, 2]**2)

    return traj


def wong_roos_interleaved_trajectory(n, nint):
    """Interleaved trajectory by Wong and Roos

    Args:
        n (int): Number of spokes
        nint (int): Number of interleaves

    Returns:
        array: Trajectory

    References:
        S. T. S. Wong and M. S. Roos, “A strategy for sampling on a sphere applied to 3D selective RF pulse design,” 
        Magn. Reson. Med., vol. 32, no. 6, pp. 778–784, 1994.
    """

    traj = np.zeros((n, 3))
    spi = int(n/nint)
    ni = np.arange(1, spi+1)

    z = -(2*ni-spi-1)/spi

    ang_vel = np.sqrt(n*np.pi/nint)*np.arcsin(z)
    for m in range(nint):
        x = np.cos(ang_vel + (2*(m+1)*np.pi)/nint) * np.sqrt(1-z**2)
        y = np.sin(ang_vel + (2*(m+1)*np.pi)/nint) * np.sqrt(1-z**2)

        traj[m*spi:(m+1)*spi, 0] = x
        traj[m*spi:(m+1)*spi, 1] = y
        traj[m*spi:(m+1)*spi, 2] = z

    return traj


def traj2points(traj, npoints, OS):
    """
    Transform spoke trajectory to point trajectory

    Args:
        traj: Trajectory with shape [nspokes, 3]
        npoints: Number of readout points along spokes
        OS: Oversampling

    Returns:
        array: Trajectory with shape [nspokes, npoints, 3]
    """

    [nspokes, ndim] = np.shape(traj)

    r = (np.arange(0, npoints))/OS
    Gx, Gy, Gz = np.meshgrid(r, np.arange(nspokes), np.arange(ndim))
    traj_p = Gx*np.transpose(np.tile(traj, [npoints, 1, 1]), [1, 0, 2])

    return traj_p


####### OLD ########

def infinite_phyllotaxis(n, nint, sf):
    traj = np.zeros((n, 3))
    spi = int(n/nint)

    i = np.arange(spi)
    phi0 = i * PHI_GOLD * fibonacciNum[sf]
    z0 = 1 - 2*i/spi
    r = 1
    g = 3-np.sqrt(5)

    for k in range(nint):
        dz = np.mod(g*k, 1)*2/spi
        z = z0 - dz
        phi = phi0 + k * PHI_GOLD

        theta = np.arccos(z)
        traj[k*spi: (k+1)*spi, 0] = r * np.sin(theta) * np.cos(phi)
        traj[k*spi: (k+1)*spi, 1] = r * np.sin(theta) * np.sin(phi)
        traj[k*spi: (k+1)*spi, 2] = r * z

    return traj


def sense_selfcalib(y, coord, oshape=None, rf=0.25, wf=0.05):
    """
    Generate SENSE maps from 3D radial data using the center of k-space.
    Applies a fermi filter to lowpass filter the data. DC filter is 
    applied in here as well.

    Based on: McKenzie CA et al. Magn Reson Med. 2002;47(3):529–38. 

    Inputs:
        - y: Radial k-space data (raw)
        - coord: Coordinates/trajectory point by point
        - oshape: Shape of output data
        - rf: Radius for fermi filter (0.25)
        - wf: Width for fermi filter (0.05)

    Returns:
        - SENSE: Sensitivity maps
    """

    [nrcv, nspokes, npts] = np.shape(y)
    dcf = dc_filter(npts)
    ff = fermi_filter(npts, rf, wf)

    # Calculate low resolution coil images
    I_coils = sigpy.nufft_adjoint(y*ff*dcf, coord, oshape=oshape)

    # Calculate SoS image
    I_rss = np.sum(np.abs(I_coils)**2, 0)**0.5

    # Make SENSE maps by dividing out PD information from the coil images
    SENSE = I_coils/I_rss

    return SENSE


def dc_filter(n):
    """
    Simple r^2 DC filter for 3D radial acquisition
    """

    r = np.linspace(0, 1, num=n)
    dcf = r**2

    return dcf


def fermi_filter(n, rf, wf):
    """
    Fermi filter for radial out data

    Inputs:
        - n: Number of points along spoke
        - rf: Filter radius
        - wf: Filter width

    Outputs:
        - filt: 1D Fermi filter
    """
    r = np.linspace(0, 1, num=n)
    filt = 1.0/(1+np.exp((r-rf)/wf))

    return filt


def rss(I_coils):
    """
    Calculate root sum of squares of coil images
    Assumes coils along 0th dimension as in Sigpy
    """
    return np.sum(np.abs(I_coils)**2, axis=0)**0.5


def TV_recon(raw, traj, lamda, maps, show_pbar=True, max_power_iter=10, max_iter=30) -> np.complex64:
    """
    Perform Total Variation reconstruction using Sigpy. 

    In the first step it calculates a simple SoS image to get correct scaling of the 
    k-space data.

    Inputs:
        - raw: k-space [nrcv, nspokes, npts]
        - traj: K-space coordinates
        - lamda: TV lambda
        - maps: SENSE maps
        - OS: Trajectory oversampling factor
        - show_pbar: Show sigpy progress bar (True)
        - max_power_iter: Itterations to calculate linop eigenvalue (10)
        - max_iter: Maximum iterations for TV loop (30)

    Returns:
        - I: Complex valued TV image 
    """
    [nrcv, nspokes, npts] = np.shape(raw)
    dcf = dc_filter(npts)

    I_int = rss(sigpy.nufft_adjoint(raw * dcf, traj))
    img_scale = 1/np.max(I_int)
    TV_app = TotalVariationRecon(y=raw * img_scale, weights=dcf, coord=traj,
                                 mps=maps, lamda=lamda, max_power_iter=max_power_iter,
                                 max_iter=max_iter, show_pbar=show_pbar)
    I = TV_app.run()

    return I


def TV_recon_timeseries(raw, traj, nint, OS=1, SENSE_rf=0.15, SENSE_wf=0.05, lamda=0.05, show_pbar_TV=True, max_power_iter=10, max_iter=30):

    [nrcv, nspokes, npts] = np.shape(raw)

    spi = int(nspokes/nint)
    SENSE_oshape = None
    for i in tqdm.tqdm(range(nint)):

        i0 = i*spi
        i1 = (i+1)*spi

        SENSE = sense_selfcalib(
            raw[:, i0:i1, :], traj[i0:i1, :, :]*OS, rf=0.15, wf=0.05, oshape=SENSE_oshape)

        I_TV = TV_recon(raw[:, i0:i1, :], traj[i0:i1, :, :]*OS, lamda=lamda,
                        maps=SENSE, show_pbar=show_pbar_TV, max_power_iter=max_power_iter, max_iter=max_iter)

        if i == 0:
            [nx, ny, nz] = np.shape(I_TV)
            TS = np.zeros((nx, ny, nz, nint), dtype='complex')
            SENSE_oshape = (nrcv, nx, ny, nz)

        TS[:, :, :, i] = I_TV

    return TS


def svd_coil_compress(raw, n_out):
    """
    Perform simple coil compression on radial k-space data
    """
    [nrcv, nspokes, npts] = np.shape(raw)
    X = np.reshape(np.transpose(raw, (1, 2, 0)), (npts*nspokes, nrcv))
    m, n = np.shape(X)

    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)

    Sigma_cc = Sigma
    Sigma_cc[6:-1] = 0
    X_cc = np.dot(U, np.diag(Sigma_cc))

    var_exp = Sigma**2/np.sum(Sigma**2)*100
    print('Variance explained: %.1f' % (np.sum(var_exp[0:n_out])))

    raw_cc = np.transpose(np.reshape(X_cc, (nspokes, npts, nrcv)), (2, 0, 1))

    return raw_cc, Sigma


def pca_cc_calc(raw, n_out):
    """
    Calculate PCA coil compression matrix A from raw data. 
    Recommended to input subset of data for coil compression for faster
    computation.

    Input:
        - raw: Raw data to calculate coil compression on
        - n_out: Number of coils to keep

    Output:
        - A: Compression matrix (complex array)
        - pca_res: Results from PCA and eigenvalue computation

    Example usage

        A, pca_res = pca_cc_calc(raw[:,:,0:32])
        raw_cc = pca_cc_apply(raw, A)
    """

    [nrcv, nspokes, npts] = np.shape(raw)

    X = np.reshape(np.transpose(raw, (1, 2, 0)), (nspokes*npts, nrcv))

    # Center data
    u = np.mean(X, axis=0)
    B = X - u

    # Covariance matrix
    C = 1/(nspokes*npts - 1) * (np.conj(B).T @ B)

    # Eigen value decomposition
    w, v = np.linalg.eig(C)

    # Pick out values to keep
    A = v[:, 0:n_out]

    # Energy preserved
    g_out = np.sum(abs(w[0:n_out]))/np.sum(abs(w))
    print('Energy preserved: %.3f' % g_out)

    pca_res = {'C': C, 'W': w, 'V': v}

    return A, pca_res


def pca_cc_apply(raw, A):
    """
    Apply coil compression matrix

    Input:
        - raw: Raw k-space data
        - A: Coil compression matrix

    Output:
        - raw_cc: Compressed raw k-space data
    """

    [nrcv, nspokes, npts] = np.shape(raw)

    X = np.reshape(np.transpose(raw, (1, 2, 0)), (nspokes*npts, nrcv))
    Xcc = X @ A

    nrcv_out = np.shape(A)[1]
    raw_cc = np.transpose(np.reshape(
        Xcc, [nspokes, npts, nrcv_out]), (2, 0, 1))

    return raw_cc
