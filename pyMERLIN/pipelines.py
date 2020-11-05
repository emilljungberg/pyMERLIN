from .reg import itk_versor3Dreg_v1, versor_reg_summary, versor_resample, reg_to_kcorr, normalise_timeseries
from .dataIO import *
from .recon import *
from .plot import *
from sigpy import nufft_adjoint
from sigpy.mri.app import TotalVariationRecon
from tqdm import tqdm
import itk


def MERLIN_v1(raw, traj, nint, TV_lambda, ds, voxel_size):
    """
    Full Reconstruction pipeline for single channel data

    Inputs:
        - raw: k-space data
        - traj: trajectory
        - nint: Number of sequential interleaves
        - TV_lambda: Lambda for TV regularisation
        - ds: Downsampling factor
        - voxel_size: Voxel size

    Outputs:
        - H: Phase correction
        - traj_corr: Corrected trajectory
    """

    # 1. Reconstruct combined data without MOCO, plot 3 plane image
    [nrcv, nspokes, npts] = np.shape(raw)
    dcf = dc_filter(npts)
    ff = fermi_filter(npts, 0.9, 0.1)

    print('1. Reconstructing combined data wo MOCO')
    I_nomoco = rss(nufft_adjoint(raw * dcf * ff, coord=traj))

    # Look at data
    plot_3plane(I_nomoco, title='Combined Recon without MOCO')

    # 2. Reconstruct single interleave to get image scaling
    spi = int(nspokes/nint)
    npts_ds = int(npts/ds)
    dcf_ds = dc_filter(npts_ds)
    ff_ds = fermi_filter(npts_ds, 0.9, 0.1)

    print('Reconstructing single interleave')
    I_int = rss(nufft_adjoint(
        raw[:, 0:spi, 0:npts_ds] * dcf_ds * ff_ds, coord=traj[0:spi, 0:npts_ds, :]))
    img_scale = 1/np.max(I_int)

    plot_3plane(I_int, title='Single Interleave')

    # 3. Reconstruct interleaves with TV
    print('Reconstructing all interleaves', flush=True)
    [nx, ny, nz] = np.shape(I_int)
    TS = np.zeros((nx, ny, nz, nint))

    # Fake SENSE maps for single channel data
    SENSE_ds = np.ones((1, nx, ny, nz))

    for i in tqdm(range(nint)):
        i0 = i*spi
        i1 = (i+1)*spi
        TV_app = TotalVariationRecon(y=raw[:, i0:i1, 0:npts_ds] * ff_ds * img_scale,
                                     weights=dcf_ds, coord=traj[i0:i1,
                                                                0:npts_ds, :],
                                     mps=SENSE_ds, lamda=TV_lambda, max_power_iter=10, max_iter=30, show_pbar=False)

        TS[:, :, :, i] = abs(TV_app.run())

    # Normalise timeseries
    TS = normalise_timeseries(TS, 0)

    # 4. Run registration
    # Create list of ITK images
    print('Creating list of ITK images')
    images = []
    nint = np.size(TS, 3)
    for i in range(nint):
        images.append(create_image(np.transpose(
            TS[:, :, :, i], (2, 1, 0)), voxel_size*ds))

    registrations = []
    regouts = []
    reg_df = []

    ImageType = type(images[0])
    MsMetricType = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType]

    # Register data
    print('Registering images', flush=True)
    for i in tqdm(range(nint)):
        reg, reg_out = itk_versor3Dreg_v1(images[0], images[i],
                                          metric='MS', verbose=False)
        reg.Update()
        registrations.append(reg)
        regouts.append(reg_out)

    names = ['Int %d' % (x+1) for x in range(np.size(TS, 3))]
    reg_df = versor_reg_summary(registrations, regouts, names, doprint=False)

    # Transform input data
    RescaleFilterType = itk.RescaleIntensityImageFilter[ImageType, ImageType]
    TS_out = np.zeros_like(TS)

    for i in range(nint):
        rescaleFilter = RescaleFilterType.New()
        rescaleFilter.SetInput(images[i])
        rescaleFilter.SetOutputMinimum(0)
        rescaleFilter.SetOutputMaximum(1000)
        rescaleFilter.Update()

        reg_img = versor_resample(
            registrations[i], rescaleFilter.GetOutput(), images[0])
        TS_out[:, :, :, i] = itk.array_from_image(reg_img)

    reg_res = {'TSreg': TS_out, 'Registrations': registrations,
               'Regouts': regouts, 'RegDf': reg_df}

    # Display results
    fig = plt.figure(figsize=(13, 3))
    x = np.arange(nint)

    for (i, l) in enumerate(['X', 'Y', 'Z']):
        fig.add_subplot(1, 3, i+1)
        plt.plot(x, reg_res['RegDf'].loc['Trans %s' %
                                         l, :], '-o', label='Between Volumes')
        plt.grid()
        plt.title('Translation in %s' % l)
        plt.xlabel('Interleave')
        plt.ylabel('Distance [mm]')
        if i == 1:
            plt.legend()

    plt.tight_layout()
    plt.show()
    fig = plt.figure(figsize=(13, 3))
    for (i, l) in enumerate(['X', 'Y', 'Z']):
        fig.add_subplot(1, 3, i+1)
        plt.plot(x, np.rad2deg(
            reg_res['RegDf'].loc['Versor %s' % l, :]), '-o', label='Still')
        plt.grid()
        plt.title('Rotation around %s' % l)
        plt.xlabel('Interleave')
        plt.ylabel('Rotation [deg]')

    plt.tight_layout()
    plt.show()

    # 5. Calculate k-space corrections
    H_all = np.zeros_like(raw)
    traj_corr_all = np.zeros_like(traj)

    print('Calculate correction factor', flush=True)
    for i in tqdm(range(nint)):
        i0 = i*spi
        i1 = (i+1)*spi
        # For nominal voxel size
        H, R, traj_rot = reg_to_kcorr(
            reg_res['Registrations'][i], traj[i0:i1, :, :], 3*[voxel_size])
        H_all[:, i0:i1, :] = H
        traj_corr_all[i0:i1, :, :] = traj_rot

    # 6. Reconstruct combined data with corrections
    print('Reconstruct data with corrections')
    I_MERLIN = sigpy.nufft_adjoint(raw*dcf*H_all*ff, traj_corr_all)

    plot_3plane(rss(I_MERLIN), 'Combined reconstruction with MERLIN')

    # Return: phase correction and corrected trajectory

    return H_all, traj_corr_all
