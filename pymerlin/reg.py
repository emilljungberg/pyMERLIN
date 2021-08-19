# -*- coding: utf-8 -*-
"""
Containes the core registration components for MERLIN. The framework builds on ITK and is heavily inspired by ANTs.
"""

import logging
import os
import pickle

import itk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataIO import create_image, read_image_h5


def sphere_mask(image_reference, radius):
    """Make spherical mask

    Args:
        image_reference (itk.Image): Input image
        radius ([type]): Mask relative radius (<1)

    Returns:
        itk.Image: Binary brain mask
    """

    img_size = image_reference.GetLargestPossibleRegion().GetSize()
    spacing = np.array(image_reference.GetSpacing())

    rx = np.linspace(-1, 1, img_size[0])
    ry = np.linspace(-1, 1, img_size[1])
    rz = np.linspace(-1, 1, img_size[2])

    [X, Y, Z] = np.meshgrid(rx, ry, rz)
    sphere_radius = np.sqrt(X**2 + Y**2 + Z**2)
    sphere = np.zeros_like(sphere_radius)
    sphere[sphere_radius < radius] = 1

    sphere_img = create_image(sphere, spacing, dtype=itk.UC)

    return sphere_img


def brain_mask(input_image, hole_radius=5, dilation=3, gauss_variance=100, gauss_max_ker=30):
    """Brain mask based on Otsu filter

    Also applies hold filling, mask dilation, and smoothing on final mask

    Args:
        input_image (itk.Image): Input image
        hole_radius (int, optional): Hole radius to fill. Defaults to 5.
        dilation (int, optional): Dilate mask by X voxels. Defaults to 3.
        gauss_variance (int, optional): Variance of smoothing filter. Defaults to 100.
        gauss_max_ker (int, optional): Max kernel width of smoothing filter. Defaults to 30.

    Returns:
        itk.Image: Brain mask
    """
    Dimension = 3
    InputPixelType = itk.template(input_image)[1][0]
    OutputPixelType = itk.UC
    InputImageType = itk.Image[InputPixelType, Dimension]
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Otsu Therholding Filter
    OtsuFilterType = itk.OtsuThresholdImageFilter[InputImageType,
                                                  OutputImageType]
    otsu_filter = OtsuFilterType.New()
    otsu_filter.SetOutsideValue(1)
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetInput(input_image)

    # Fill Holes
    FillFilterType = itk.VotingBinaryHoleFillingImageFilter[OutputImageType, OutputImageType]
    fill_filter = FillFilterType.New()
    fill_filter.SetRadius([hole_radius, hole_radius, hole_radius])
    fill_filter.SetBackgroundValue(0)
    fill_filter.SetForegroundValue(1)
    fill_filter.SetMajorityThreshold(10)
    fill_filter.SetInput(otsu_filter.GetOutput())

    # Dilate mask
    StructuringElementType = itk.FlatStructuringElement[Dimension]
    DilateFilterType = itk.BinaryDilateImageFilter[OutputImageType,
                                                   OutputImageType, StructuringElementType]
    binaryDilate = DilateFilterType.New()
    structuringElement = StructuringElementType()
    structuringElement.SetRadius(dilation)
    binaryDilate.SetKernel(structuringElement)
    binaryDilate.SetDilateValue(1)
    binaryDilate.SetInput(fill_filter.GetOutput())

    # Cast back to float image
    CastFilterType = itk.RescaleIntensityImageFilter[OutputImageType, InputImageType]
    cast_filter = CastFilterType.New()
    cast_filter.SetOutputMinimum(0)
    cast_filter.SetOutputMaximum(1)
    cast_filter.SetInput(binaryDilate.GetOutput())

    # Smooth data
    SmoothingFilterType = itk.DiscreteGaussianImageFilter[InputImageType, InputImageType]
    smoothing_filter = SmoothingFilterType.New()
    smoothing_filter.SetVariance(gauss_variance)
    smoothing_filter.SetMaximumKernelWidth(gauss_max_ker)
    smoothing_filter.SetInput(cast_filter.GetOutput())

    # Final update
    smoothing_filter.Update()

    threshold = otsu_filter.GetThreshold()
    print("Otsu Threshold: {}".format(threshold))
    output = smoothing_filter.GetOutput()

    return output


def otsu_filter(image):
    """Applies an Otsu filter

    Args:
        image (itk.Image): Input image

    Returns:
        itk.OtsuThresholdImageFilter: Filter
    """

    print("Performing Otsu thresholding")
    OtsuOutImageType = itk.Image[itk.UC, 3]
    filt = itk.OtsuThresholdImageFilter[type(image),
                                        OtsuOutImageType].New()
    filt.SetInput(image)
    filt.Update()

    return filt


def versor_reg_summary(registrations, reg_outs, names=None, doprint=True, show_legend=True):
    """Summarise results from one or more versor registration experiments


    Args:
        registrations (list): List of registration objects
        reg_outs (list): List of dictionaries of registration outputs
        names (list, optional): Labels for each registration. Defaults to None.
        doprint (bool, optional): Print output. Defaults to True.
        show_legend (bool, optional): Show plot legend. Defaults to True.

    Returns:
        pandas.DataFrame: Summary of registrations
    """

    df_dict = {}
    index = ['Trans X', 'Trans Y', 'Trans Z',
             'Versor X', 'Versor Y', 'Versor Z',
             'Iterations', 'Metric Value']
    if doprint:
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12, 8))

    if not names:
        names = ['Int %d' % x for x in range(len(registrations))]

    for (reg, reg_out, name) in zip(registrations, reg_outs, names):
        # Examine the result
        transform = reg.GetTransform()
        optimizer = reg.GetOptimizer()
        final_parameters = transform.GetParameters()

        versorX = final_parameters[0]
        versorY = final_parameters[1]
        versorZ = final_parameters[2]
        transX = final_parameters[3]
        transY = final_parameters[4]
        transZ = final_parameters[5]
        nits = optimizer.GetCurrentIteration()
        best_val = optimizer.GetValue()

        # Summarise data and store in dictionary
        reg_data = [transX, transY, transZ,
                    versorX, versorY, versorZ,
                    nits, best_val]
        df_dict[name] = reg_data

        # Creat plots
        if doprint:
            ax = axes[0, 0]
            ax.plot(reg_out['cv'], '-o')
            ax.set_ylabel('')
            ax.set_title('Optimizer Value')
            ax.grid('on')

            ax = axes[0, 1]
            ax.plot(reg_out['lrr'], '-o')
            ax.set_ylabel('')
            ax.set_title('Learning Rate Relaxation')
            ax.grid('on')

            ax = axes[0, 2]
            ax.plot(reg_out['sl'], '-o', label=name)
            ax.set_ylabel('')
            ax.set_title('Step Length')
            ax.grid('on')
            if show_legend:
                ax.legend()

            ax = axes[1, 0]
            ax.plot(reg_out['tX'], '-o')
            ax.set_ylabel('[mm]')
            ax.set_title('Translation X')
            ax.grid('on')

            ax = axes[1, 1]
            ax.plot(reg_out['tY'], '-o')
            ax.set_ylabel('[mm]')
            ax.set_title('Translation Y')
            ax.grid('on')

            ax = axes[1, 2]
            ax.plot(reg_out['tZ'], '-o')
            ax.set_ylabel('[mm]')
            ax.set_title('Translation Z')
            ax.grid('on')

            ax = axes[2, 0]
            ax.plot(reg_out['vX'], '-o')
            ax.set_xlabel('Itteration')
            ax.set_ylabel('')
            ax.set_title('Versor X')
            ax.grid('on')

            ax = axes[2, 1]
            ax.plot(reg_out['vY'], '-o')
            ax.set_xlabel('Itteration')
            ax.set_ylabel('')
            ax.set_title('Versor Y')
            ax.grid('on')

            ax = axes[2, 2]
            ax.plot(reg_out['vZ'], '-o')
            ax.set_xlabel('Itteration')
            ax.set_ylabel('')
            ax.set_title('Versor Z')
            ax.grid('on')

    if doprint:
        plt.tight_layout()
        plt.show()

    # Create Dataframe with output data
    df = pd.DataFrame(df_dict, index=index)

    # Determine if running in notebook or shell to get right print function
    env = os.environ
    program = os.path.basename(env['_'])

    if doprint:
        print(df)

    return df


def versor_watcher(reg_out, optimizer):
    """Logging for registration

    Args:
        reg_out (dict): Structure for logging registration
        optimizer (itk.RegularStepGradientDescentOptimizerv4): Optimizer object

    Returns:
        function: Logging function
    """

    logging.debug("{:s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s}".format(
        'Itt', 'Value', 'vX [deg]', 'vY [deg]', 'vZ [deg]', 'tX [mm]', 'tY [mm]', 'tZ [mm]'))

    def opt_watcher():
        cv = optimizer.GetValue()
        cpos = np.array(optimizer.GetCurrentPosition())
        cit = optimizer.GetCurrentIteration()
        lrr = optimizer.GetCurrentLearningRateRelaxation()
        sl = optimizer.GetCurrentStepLength()

        # Store logged values
        reg_out['cv'].append(cv)
        reg_out['vX'].append(np.rad2deg(cpos[0]))
        reg_out['vY'].append(np.rad2deg(cpos[1]))
        reg_out['vZ'].append(np.rad2deg(cpos[2]))
        reg_out['tX'].append(cpos[3])
        reg_out['tY'].append(cpos[4])
        reg_out['tZ'].append(cpos[5])
        reg_out['sl'].append(sl)
        reg_out['lrr'].append(lrr)

        logging.debug("{:d} \t {:6.5f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f}".format(
            cit, cv, np.rad2deg(cpos[0]), np.rad2deg(cpos[1]), np.rad2deg(cpos[2]), cpos[3], cpos[4], cpos[5]))

    return opt_watcher


def winsorize_image(image, p_low, p_high):
    """Applies winsorize filter to image

    Args:
        image (itk.Image): Input image
        p_low (float): Lower percentile
        p_high (float): Upper percentile

    Returns:
        itk.ThresholdImageFilter: Threshold filter
    """

    Dimension = 3
    PixelType = itk.template(image)[1][0]
    ImageType = itk.Image[PixelType, Dimension]

    # Histogram
    nbins = 1000  # Allows 0.001 precision
    hist_filt = itk.ImageToHistogramFilter[ImageType].New()
    hist_filt.SetInput(image)
    hist_filt.SetAutoMinimumMaximum(True)
    hist_filt.SetHistogramSize([nbins])
    hist_filt.Update()
    hist = hist_filt.GetOutput()

    low_lim = hist.Quantile(0, p_low)
    high_lim = hist.Quantile(0, p_high)

    filt = itk.ThresholdImageFilter[ImageType].New()
    filt.SetInput(image)
    filt.ThresholdBelow(low_lim)
    filt.ThresholdAbove(high_lim)
    filt.ThresholdOutside(low_lim, high_lim)

    return filt


def threshold_image(image, low_lim):
    """Threshold image at given value

    Args:
        image (itk.Image): Input image
        low_lim (float): Lower threshold

    Returns:
        itk.Image: Thresholded image
    """

    Dimension = 3
    PixelType = itk.template(image)[1][0]
    ImageType = itk.Image[PixelType, Dimension]

    thresh_filt = itk.ThresholdImageFilter[ImageType].New()
    thresh_filt.ThresholdBelow(float(low_lim))
    thresh_filt.SetOutsideValue(0)
    thresh_filt.SetInput(image)
    thresh_filt.Update()

    return thresh_filt.GetOutput()


def resample_image(registration, moving_image, fixed_image):
    """Resample image with registration parameters

    Args:
        registration (itk.ImageRegistrationMethodv4): Registration object
        moving_image (itk.Image): Moving image
        fixed_image (itk.Image): Fixed image

    Returns:
        itk.ResampleImageFilter: Resampler filter
    """
    logging.info("Resampling moving image")
    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()

    TransformType = itk.VersorRigid3DTransform[itk.D]
    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(
        registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(final_parameters)

    ResampleFilterType = itk.ResampleImageFilter[type(moving_image),
                                                 type(moving_image)]
    resampler = ResampleFilterType.New()
    resampler.SetTransform(finalTransform)
    resampler.SetInput(moving_image)

    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    return resampler


def get_versor_factors(registration):
    """Calculate correction factors from Versor object

    Args:
        registration (itk.ImageRegistrationMethodv4): Registration object

    Returns:
        dict: Correction factors
    """

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()

    TransformType = itk.VersorRigid3DTransform[itk.D]
    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(
        registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(final_parameters)

    matrix = convert_itk_matrix(finalTransform.GetMatrix())
    offset = np.array(finalTransform.GetOffset())
    regParameters = registration.GetOutput().Get().GetParameters()

    corrections = {'R': matrix,
                   'rx': regParameters[0],
                   'ry': regParameters[1],
                   'rz': regParameters[2],
                   'dx': regParameters[3],
                   'dy': regParameters[4],
                   'dz': regParameters[5]
                   }

    return corrections


def setup_optimizer(PixelType, opt_range, relax_factor, nit=250, learning_rate=0.1, convergence_window_size=10, convergence_value=1E-6):
    """Setup optimizer object

    Args:
        PixelType (itkCType): ITK pixel type
        opt_range (list): Range for optimizer
        relax_factor (float): Relaxation factor
        nit (int, optional): Number of iterations. Defaults to 250.
        learning_rate (float, optional): Optimizer learning rate. Defaults to 0.1.
        convergence_window_size (int, optional): Number of points to use in evaluating convergence. Defaults to 10.
        convergence_value ([type], optional): Value at which convergence is reached. Defaults to 1E-6.

    Returns:
        itk.RegularStepGradientDescentOptimizerv4: Optimizer object
    """

    logging.info("Initialising Regular Step Gradient Descent Optimizer")
    optimizer = itk.RegularStepGradientDescentOptimizerv4[PixelType].New()
    OptimizerScalesType = itk.OptimizerParameters[PixelType]
    # optimizerScales = OptimizerScalesType(
    #     initialTransform.GetNumberOfParameters())
    optimizerScales = OptimizerScalesType(6)

    # Set scales <- Not sure about this part
    rotationScale = 1.0/np.deg2rad(opt_range[0])
    translationScale = 1.0/opt_range[1]
    optimizerScales[0] = rotationScale
    optimizerScales[1] = rotationScale
    optimizerScales[2] = rotationScale
    optimizerScales[3] = translationScale
    optimizerScales[4] = translationScale
    optimizerScales[5] = translationScale
    optimizer.SetScales(optimizerScales)

    logging.info("Setting up optimizer")
    logging.info("Rot/Trans scales: {}/{}".format(opt_range[0], opt_range[1]))
    logging.info("Number of itterations: %d" % nit)
    logging.info("Learning rate: %.2f" % learning_rate)
    logging.info("Relaxation factor: %.2f" % relax_factor)
    logging.info("Convergence window size: %d" % convergence_window_size)
    logging.info("Convergence value: %.2f" % convergence_value)

    optimizer.SetNumberOfIterations(nit)
    optimizer.SetLearningRate(learning_rate)          # Default in ANTs
    optimizer.SetRelaxationFactor(relax_factor)
    optimizer.SetConvergenceWindowSize(convergence_window_size)
    optimizer.SetMinimumConvergenceValue(convergence_value)

    return optimizer


def ants_pyramid(fixed_image_fname, moving_image_fname,
                 moco_output_name=None, fixed_output_name=None,
                 fixed_mask_radius=None,
                 reg_par_name=None,
                 iteration_log_fname=None,
                 opt_range=[10, 30],
                 relax_factor=0.5,
                 winsorize=[0.005, 0.995],
                 threshold=None,
                 sigmas=[2, 1, 0], shrink=[4, 2, 1],
                 metric='MS',
                 verbose=2):
    """Multi-scale rigid body registration

    ITK registration framework inspired by ANTs which performs a multi-scale 3D versor registratio between two 3D volumes. The input data is provided as .h5 image files. 

    Prior to registration a number of filters can be applied

        - Winsorize (not recommended)
        - Thresholding (recommended)

    Args:
        fixed_image_fname (str): Fixed file (.h5 file)
        moving_image_fname (str): Moving file (.h5 file)
        moco_output_name (str, optional): Output moco image as nifti. Defaults to None.
        fixed_output_name (str, optional): Output fixed image as nifti. Defaults to None.
        fixed_mask_radius (int, optional): Radius of spherical mask for fixed image. Defaults to None.
        reg_par_name (str, optional): Name of output parameter file. Defaults to None.
        iteration_log_fname (str, optional): Name for output log file. Defaults to None.
        opt_range (list, optional): Expected range of motion [deg,mm]. Defaults to [10, 30].
        relax_factor (float, optional): Relaxation factor for optimizer. Defaults to 0.5.
        winsorize (list, optional): Limits for winsorize filter. Defaults to [0.005, 0.995].
        threshold (float, optional): Lower value for threshold filter . Defaults to None.
        sigmas (list, optional): Smoothing sigmas for registration. Defaults to [2, 1, 0].
        shrink (list, optional): Shring factors for registration. Defaults to [4, 2, 1].
        metric (str, optional): Image metric for registrationn (MI/MS). Defaults to 'MS'.
        verbose (int, optional): Level of debugging (0/1/2). Defaults to 2.

    Returns:
        (itk.ImageRegistrationMethodv4, dict, str): Registration object, Registration history, Name of output file with correction factors
    """

    # Logging
    log_level = {0: None, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=log_level[verbose], datefmt="%I:%M:%S")

    # Global settings. 3D data with itk.D type
    PixelType = itk.D
    ImageType = itk.Image[PixelType, 3]

    # Validate inputs
    if len(sigmas) != len(shrink):
        logging.error("Sigma and Shrink arrays not the same length")
        raise Exception("Sigma and Shrink arrays must be same length")

    # Read in data
    logging.info("Reading fixed image: {}".format(fixed_image_fname))
    data_fix, spacing_fix = read_image_h5(fixed_image_fname)
    logging.info("Reading moving image: {}".format(moving_image_fname))
    data_move, spacing_move = read_image_h5(moving_image_fname)
    fixed_image = create_image(data_fix, spacing_fix, dtype=PixelType)
    moving_image = create_image(data_move, spacing_move, dtype=PixelType)

    # Winsorize filter
    if winsorize:
        logging.info("Winsorising images")
        fixed_win_filter = winsorize_image(
            fixed_image, winsorize[0], winsorize[1])
        moving_win_filter = winsorize_image(
            moving_image, winsorize[0], winsorize[1])

        fixed_image = fixed_win_filter.GetOutput()
        moving_image = moving_win_filter.GetOutput()

    if threshold == 'otsu':
        logging.info("Calculating Otsu filter")
        filt = otsu_filter(fixed_image)
        otsu_threshold = filt.GetThreshold()
        logging.info(
            "Applying thresholding at Otsu threshold of {}".format(otsu_threshold))
        fixed_image = threshold_image(fixed_image, otsu_threshold)
        moving_image = threshold_image(moving_image, otsu_threshold)

    elif threshold is not None:
        logging.info("Thresholding images at {}".format(threshold))
        fixed_image = threshold_image(fixed_image, threshold)
        moving_image = threshold_image(moving_image, threshold)

    # Setup image metric
    if metric == 'MI':
        nbins = 16
        logging.info(
            "Using Mattes Mutual Information image metric with {} bins".format(nbins))
        metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType,
                                                                 ImageType].New()
        metric.SetNumberOfHistogramBins(nbins)
        metric.SetUseMovingImageGradientFilter(False)
        metric.SetUseFixedImageGradientFilter(False)
    else:
        logging.info("Using Mean Squares image metric")
        metric = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType].New(
        )

    # Setup versor transform
    logging.info("Initialising Versor Rigid 3D Transform")
    TransformType = itk.VersorRigid3DTransform[PixelType]
    TransformInitializerType = itk.CenteredTransformInitializer[TransformType,
                                                                ImageType, ImageType]

    initialTransform = TransformType.New()
    initializer = TransformInitializerType.New()
    initializer.SetTransform(initialTransform)
    initializer.SetFixedImage(fixed_image)
    initializer.SetMovingImage(moving_image)
    initializer.InitializeTransform()

    VersorType = itk.Versor[itk.D]
    VectorType = itk.Vector[itk.D, 3]
    rotation = VersorType()
    axis = VectorType()
    axis[0] = 0
    axis[1] = 0
    axis[2] = 1
    angle = 0
    rotation.Set(axis, angle)
    initialTransform.SetRotation(rotation)

    # Setup optimizer
    optimizer = setup_optimizer(PixelType, opt_range, relax_factor)

    # Setup registration
    registration = itk.ImageRegistrationMethodv4[ImageType,
                                                 ImageType].New()
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetInitialTransform(initialTransform)

    # One level registration without shrinking and smoothing
    logging.info("Smoothing sigmas: {}".format(sigmas))
    logging.info("Shrink factors: {}".format(shrink))
    numberOfLevels = len(sigmas)
    shrinkFactorsPerLevel = itk.Array[itk.F](numberOfLevels)
    smoothingSigmasPerLevel = itk.Array[itk.F](numberOfLevels)

    for i in range(numberOfLevels):
        shrinkFactorsPerLevel[i] = shrink[i]
        smoothingSigmasPerLevel[i] = sigmas[i]

    registration.SetNumberOfLevels(numberOfLevels)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel)
    registration.SetShrinkFactorsPerLevel(shrinkFactorsPerLevel)

    if fixed_mask_radius:
        logging.info("Creating fixed space spherical mask with radius: {}".format(
            fixed_mask_radius))
        MaskType = itk.ImageMaskSpatialObject[3]
        spatial_mask = MaskType.New()
        mask = sphere_mask(fixed_image, fixed_mask_radius)
        spatial_mask.SetImage(mask)
        spatial_mask.Update()
        metric.SetFixedImageMask(spatial_mask)

    # Watch the itteration events
    reg_out = {'cv': [], 'tX': [], 'tY': [], 'tZ': [],
               'vX': [], 'vY': [], 'vZ': [], 'sl': [], 'lrr': []}

    logging.info("Running Registration")
    wf = versor_watcher(reg_out, optimizer)
    optimizer.AddObserver(itk.IterationEvent(), wf)

    # --> Run registration
    registration.Update()

    # Correction factors
    corrections = get_versor_factors(registration)
    logging.info("Estimated parameters")
    logging.info("Rotation: (%.2f, %.2f, %.2f) deg" %
                 (np.rad2deg(corrections['rx']),
                  np.rad2deg(corrections['ry']),
                  np.rad2deg(corrections['rz'])))
    logging.info("Translation: (%.2f, %.2f, %.2f) mm" %
                 (corrections['dx'],
                  corrections['dy'],
                  corrections['dz']))

    # Resample moving data
    resampler = resample_image(registration, moving_image, fixed_image)

    # Write output
    if moco_output_name:
        logging.info(
            "Writing moco output image to: {}".format(moco_output_name))
        writer = itk.ImageFileWriter[ImageType].New()
        writer.SetFileName(moco_output_name)
        writer.SetInput(resampler.GetOutput())
        writer.Update()

    if fixed_output_name:
        logging.info(
            "Writing reference imgae to: {}".format(fixed_output_name))
        writer = itk.ImageFileWriter[ImageType].New()
        writer.SetFileName(fixed_output_name)
        writer.SetInput(fixed_image)
        writer.Update()

    if iteration_log_fname:
        logging.info("Writing iteration log to: %s" % iteration_log_fname)
        pickle.dump(reg_out, open(iteration_log_fname, "wb"))

    if not reg_par_name:
        fix_bname = os.path.splitext(os.path.basename(fixed_image_fname))
        move_bname = os.path.splitext(os.path.basename(moving_image_fname))
        reg_par_name = "%s_2_%s_reg.p" % (move_bname, fix_bname)

    logging.info("Writing registration parameters to: %s" % reg_par_name)
    pickle.dump(corrections, open(reg_par_name, "wb"))

    return registration, reg_out, reg_par_name


def histogram_threshold_estimator(img, plot=False, nbins=200):
    """Estimate background intensity using histogram.

    Initially used to reduce streaking in background but found to make little difference.

    Args:
        img (np.array): Image
        plot (bool, optional): Plot result. Defaults to False.
        nbins (int, optional): Number of histogram bins. Defaults to 200.
    """
    def smooth(x, window_len=10, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also: 

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

        -> Obtained from the scipy cookbook at: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        Modified to use np instead of numpy
        """

        if x.ndim != 1:
            raise(ValueError, "smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise(ValueError, "Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.'+window+'(window_len)')

        y = np.convolve(w/w.sum(), s, mode='valid')

        return y[int(window_len/2-1):-int(window_len/2)]

    y, x = np.histogram(abs(img.flatten()), bins=nbins)
    x = (x[1:]+x[:-1])/2
    y = smooth(y)

    dx = (x[1:]+x[:-1])/2
    dx2 = (dx[1:]+dx[:-1])/2
    dy = np.diff(y)
    dy2 = np.diff(smooth(dy))

    # Peak of histogram
    imax = np.argmax(y)

    # Find max of second derivative after this peak
    dy2max = np.argmax(dy2[imax:])
    thr = int(dx2[imax+dy2max])

    if plot:
        plt.figure()
        plt.plot(x, y/max(y), label='H')
        plt.plot(dx, dy/np.max(dy), label='dH/dx')
        ldy2 = plt.plot(dx2, dy2/max(abs(dy2)), label=r'$dH^2/dx^2$')
        plt.axis([0, 1500, -1, 1])

        thr = int(dx2[imax+dy2max])
        plt.plot([dx2[imax+dy2max], dx2[imax+dy2max]], [-1, 1], '--',
                 color=ldy2[0].get_color(), label='Thr=%d' % thr)
        plt.legend()
        plt.show()

    return thr

    #################################################################
    # Legacy functions
    #################################################################


def versor_resample(registration, moving_image, fixed_image):

    Dimension = 3
    PixelType = itk.D
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()

    TransformType = itk.VersorRigid3DTransform[itk.D]
    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(
        registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(final_parameters)

    ResampleFilterType = itk.ResampleImageFilter[MovingImageType,
                                                 FixedImageType]
    resampler = ResampleFilterType.New()
    resampler.SetTransform(finalTransform)
    resampler.SetInput(moving_image)

    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    return resampler.GetOutput()


def make_opt_par():
    """Creates dictionary with default registration parameters.

    Recommendations and interpretation of the different parameters
    - Metric: Default is MS (mean squares)

    - Opt range: Expected range of motion in deg and mm

    - Learning rate: The initial step size in the opimiser. Too large and it will be unstable. Too small and it might not reach the minimum

    - Relaxation factor: The fraction by which the step size is reduced every time the optimiser changes direction. Too small value will reduce the step size too quickly and can risk local minima. Too large and the optimiser might need too many itterations. For noisy data the optimiser might change a lot and a higher value might be good.
    """
    D = {'metric': 'MS',
         'opt_range': [10, 30],
         'learning_rate': 0.2,
         'min_step_length': 0.001,
         'relax_factor': 0.6}

    return D


def convert_itk_matrix(m):
    """Create numpy matrix from itk 3x3 matrix object.

    Args:
        m (itk matrix): Input matrix

    Returns:
        np.array: Numpy 3x3 matrix
    """
    np_m = np.ndarray([3, 3])
    for i in range(3):
        for j in range(3):
            np_m[i, j] = m(i, j)

    return np_m
