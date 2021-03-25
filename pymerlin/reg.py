import h5py
import itk
import logging
import os
import pickle
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML

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
        if program == 'jupyter':
            # Use HTML print in Jupyter Notebook
            display(HTML(df.to_html()))
        else:
            # All other cases use default print method
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


#################################################################
# Legacy functions
#################################################################


def itk_versor3Dreg_v2(fixed_image, moving_image, verbose=True):

    # Input specifications
    Dimension = 3
    PixelType = itk.D
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    TransformType = itk.VersorRigid3DTransform[itk.D]
    OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
    MetricType = itk.MeanSquaresImageToImageMetricv4[FixedImageType,
                                                     MovingImageType]
    RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType,
                                                     MovingImageType]

    metric = MetricType.New()
    optimizer = OptimizerType.New()
    registration = RegistrationType.New()

    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)

    initialTransform = TransformType.New()
    # Center the transform
    fixedSpacing = fixed_image.GetSpacing()
    fixedOrigin = fixed_image.GetOrigin()
    fixedRegion = fixed_image.GetLargestPossibleRegion()
    fixedSize = fixed_image.GetSize()

    centerFixed = [0, 0]
    centerFixed[0] = fixedOrigin[0] + fixedSpacing[0]*fixedSize[0]/2
    centerFixed[1] = fixedOrigin[1] + fixedSpacing[1]*fixedSize[1]/2

    movingSpacing = moving_image.GetSpacing()
    movingOrigin = moving_image.GetOrigin()
    movingRegion = moving_image.GetLargestPossibleRegion()
    movingSize = moving_image.GetSize()

    centerMoving = [0, 0]
    centerMoving[0] = movingOrigin[0] + movingSpacing[0]*movingSize[0]/2
    centerMoving[1] = movingOrigin[1] + movingSpacing[1]*movingSize[1]/2

    initialTransform.SetCenter(centerFixed)
    initialTransform.SetTranslation(centerMoving - centerFixed)
    initialTransform.SetAngle(0.0)

    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)

    TransformInitializerType = itk.CenteredTransformInitializer[TransformType,
                                                                FixedImageType, MovingImageType]
    initializer = TransformInitializerType.New()

    initializer.SetTransform(initialTransform)
    initializer.SetFixedImage(fixed_image)
    initializer.SetMovingImage(moving_image)
    # initializer.MomentsOn()
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

    registration.SetInitialTransform(initialTransform)

    OptimizerScalesType = itk.OptimizerParameters[itk.D]
    optimizerScales = OptimizerScalesType(
        initialTransform.GetNumberOfParameters())

    translationScale = 1.0/1000.0
    optimizerScales[0] = 1.0
    optimizerScales[1] = 1.0
    optimizerScales[2] = 1.0
    optimizerScales[3] = translationScale
    optimizerScales[4] = translationScale
    optimizerScales[5] = translationScale
    optimizer.SetScales(optimizerScales)

    ### Specifications of the Optimizer ###
    optimizer.SetNumberOfIterations(200)
    optimizer.SetLearningRate(0.2)
    # optimizer.SetRelaxationFactor(0.6)
    optimizer.SetMinimumStepLength(0.001)
    # optimizer.SetMaximumStepSizeInPhysicalUnits(1.3)
    optimizer.SetReturnBestParametersAndValue(True)

    # One level registration without shrinking and smoothing
    numberOfLevels = 1
    shrinkFactorsPerLevel = itk.Array[itk.F](1)
    shrinkFactorsPerLevel[0] = 1

    smoothingSigmasPerLevel = itk.Array[itk.F](1)
    smoothingSigmasPerLevel[0] = 0

    registration.SetNumberOfLevels(numberOfLevels)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])

    # Watch the itteration events
    reg_out = {'cv': [], 'tX': [], 'tY': [], 'tZ': [],
               'vX': [], 'vY': [], 'vZ': [], 'sl': [], 'lrr': []}

    wf = versor_watcher(reg_out, optimizer)
    optimizer.AddObserver(itk.IterationEvent(), wf)

    return registration, optimizer, reg_out


def itk_versor3Dreg_v1(fixed_image, moving_image, fixed_mask=None, metric='MS', opt_range=[10, 30], learning_rate=1,
                       min_step_length=0.001, relax_factor=0.5, verbose=True):
    """
    Perform 3D versor registration between two images

    Inputs:
        - fixed_image: Fixed image
        - moving_image: Moving image
        - MetricType: Image metric (MS)
        - opt_range: Range of expected motion [deg, mm] (10, 30)

    Outputs:
        - registration: Registration object
        - reg_out: Registration debug stuff
    """

    # Input specifications
    Dimension = 3
    PixelType = itk.template(fixed_image)[1][0]
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    TransformType = itk.VersorRigid3DTransform[PixelType]
    OptimizerType = itk.RegularStepGradientDescentOptimizerv4[PixelType]

    if metric == 'MI':
        MetricType = itk.MattesMutualInformationImageToImageMetricv4[FixedImageType,
                                                                     MovingImageType]
    else:
        MetricType = itk.MeanSquaresImageToImageMetricv4[FixedImageType,
                                                         MovingImageType]

    RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType,
                                                     MovingImageType]

    metric = MetricType.New()
    optimizer = OptimizerType.New()
    registration = RegistrationType.New()

    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)

    initialTransform = TransformType.New()

    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)

    TransformInitializerType = itk.CenteredTransformInitializer[TransformType,
                                                                FixedImageType, MovingImageType]
    initializer = TransformInitializerType.New()

    initializer.SetTransform(initialTransform)
    initializer.SetFixedImage(fixed_image)
    initializer.SetMovingImage(moving_image)
    # initializer.MomentsOn()
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

    registration.SetInitialTransform(initialTransform)

    OptimizerScalesType = itk.OptimizerParameters[PixelType]
    optimizerScales = OptimizerScalesType(
        initialTransform.GetNumberOfParameters())

    # Set scales
    rotationScale = 1.0/np.deg2rad(opt_range[0])
    translationScale = 1.0/opt_range[1]
    optimizerScales[0] = rotationScale
    optimizerScales[1] = rotationScale
    optimizerScales[2] = rotationScale
    optimizerScales[3] = translationScale
    optimizerScales[4] = translationScale
    optimizerScales[5] = translationScale
    optimizer.SetScales(optimizerScales)

    ### Specifications of the Optimizer ###
    optimizer.SetNumberOfIterations(200)
    optimizer.SetLearningRate(learning_rate)
    optimizer.SetRelaxationFactor(relax_factor)
    optimizer.SetMinimumStepLength(min_step_length)
    # optimizer.SetMaximumStepSizeInPhysicalUnits(1.3)
    optimizer.SetReturnBestParametersAndValue(True)

    # One level registration without shrinking and smoothing
    numberOfLevels = 1
    shrinkFactorsPerLevel = itk.Array[itk.F](1)
    shrinkFactorsPerLevel[0] = 1

    smoothingSigmasPerLevel = itk.Array[itk.F](1)
    smoothingSigmasPerLevel[0] = 0

    registration.SetNumberOfLevels(numberOfLevels)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])

    if fixed_mask:
        dimension = 3
        MaskInputPixelType = itk.template(fixed_mask)[1][0]
        MaskOutputPixelType = itk.UC
        MaskInputImageType = itk.Image[MaskInputPixelType, Dimension]
        MaskOutputImageType = itk.Image[MaskOutputPixelType, Dimension]

        CastFilterType = itk.RescaleIntensityImageFilter[MaskInputImageType,
                                                         MaskOutputImageType]
        mask_cast = CastFilterType.New()
        mask_cast.SetOutputMinimum(0)
        mask_cast.SetOutputMaximum(1)
        mask_cast.SetInput(fixed_mask)
        mask_cast.Update()
        mask_out = mask_cast.GetOutput()

        MaskType = itk.ImageMaskSpatialObject[3]
        spatial_mask = MaskType.New()
        spatial_mask.SetImage(mask_out)
        spatial_mask.Update()

    # Watch the itteration events
    reg_out = {'cv': [], 'tX': [], 'tY': [], 'tZ': [],
               'vX': [], 'vY': [], 'vZ': [], 'sl': [], 'lrr': []}

    wf = versor_watcher(reg_out, optimizer)
    optimizer.AddObserver(itk.IterationEvent(), wf)

    return registration, reg_out


def itk_versor3Dreg_v3(fixed_image, moving_image, fixed_mask=None, metric='MI', opt_range=[10, 30], learning_rate=1,
                       min_step_length=0.001, relax_factor=0.5, verbose=True):
    """
    Perform 3D versor registration between two images

    Inputs:
        - fixed_image: Fixed image
        - moving_image: Moving image
        - MetricType: Image metric (MS)
        - opt_range: Range of expected motion [deg, mm] (10, 30)

    Outputs:
        - registration: Registration object
        - reg_out: Registration debug stuff
    """

    # Input specifications
    Dimension = 3
    PixelType = itk.template(fixed_image)[1][0]
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    if metric == 'MI':
        metric = itk.MattesMutualInformationImageToImageMetricv4[FixedImageType,
                                                                 MovingImageType].New()
        metric.SetNumberOfHistogramBins(32)
    else:
        metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType,
                                                     MovingImageType].New()

    optimizer = itk.RegularStepGradientDescentOptimizerv4[PixelType].New()

    registration = itk.ImageRegistrationMethodv4[FixedImageType,
                                                 MovingImageType].New()
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)

    TransformType = itk.VersorRigid3DTransform[PixelType]
    initialTransform = TransformType.New()

    TransformInitializerType = itk.CenteredTransformInitializer[TransformType,
                                                                FixedImageType, MovingImageType]
    initializer = TransformInitializerType.New()

    initializer.SetTransform(initialTransform)
    initializer.SetFixedImage(fixed_image)
    initializer.SetMovingImage(moving_image)
    # initializer.MomentsOn()
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

    registration.SetInitialTransform(initialTransform)

    OptimizerScalesType = itk.OptimizerParameters[PixelType]
    optimizerScales = OptimizerScalesType(
        initialTransform.GetNumberOfParameters())

    # Set scales
    rotationScale = 1.0/np.deg2rad(opt_range[0])
    translationScale = 1.0/opt_range[1]
    optimizerScales[0] = rotationScale
    optimizerScales[1] = rotationScale
    optimizerScales[2] = rotationScale
    optimizerScales[3] = translationScale
    optimizerScales[4] = translationScale
    optimizerScales[5] = translationScale
    optimizer.SetScales(optimizerScales)

    ### Specifications of the Optimizer ###
    optimizer.SetNumberOfIterations(100)
    optimizer.SetLearningRate(learning_rate)
    optimizer.SetRelaxationFactor(relax_factor)
    optimizer.SetMinimumStepLength(min_step_length)
    # optimizer.SetMaximumStepSizeInPhysicalUnits(1.3)
    optimizer.SetReturnBestParametersAndValue(True)

    # One level registration without shrinking and smoothing
    numberOfLevels = 4
    shrinkFactorsPerLevel = itk.Array[itk.F](numberOfLevels)
    shrinkFactorsPerLevel[0] = 2
    shrinkFactorsPerLevel[1] = 1

    smoothingSigmasPerLevel = itk.Array[itk.F](numberOfLevels)
    smoothingSigmasPerLevel[0] = 1
    smoothingSigmasPerLevel[0] = 0

    registration.SetNumberOfLevels(numberOfLevels)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel)
    registration.SetShrinkFactorsPerLevel(shrinkFactorsPerLevel)

    if fixed_mask:
        dimension = 3
        MaskInputPixelType = itk.template(fixed_mask)[1][0]
        MaskOutputPixelType = itk.UC
        MaskInputImageType = itk.Image[MaskInputPixelType, Dimension]
        MaskOutputImageType = itk.Image[MaskOutputPixelType, Dimension]

        CastFilterType = itk.RescaleIntensityImageFilter[MaskInputImageType,
                                                         MaskOutputImageType]
        mask_cast = CastFilterType.New()
        mask_cast.SetOutputMinimum(0)
        mask_cast.SetOutputMaximum(1)
        mask_cast.SetInput(fixed_mask)
        mask_cast.Update()
        mask_out = mask_cast.GetOutput()

        MaskType = itk.ImageMaskSpatialObject[3]
        spatial_mask = MaskType.New()
        spatial_mask.SetImage(mask_out)
        spatial_mask.Update()

    # Watch the itteration events
    reg_out = {'cv': [], 'tX': [], 'tY': [], 'tZ': [],
               'vX': [], 'vY': [], 'vZ': [], 'sl': [], 'lrr': []}

    wf = versor_watcher(reg_out, optimizer)
    optimizer.AddObserver(itk.IterationEvent(), wf)

    return registration, reg_out


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


def merlin_ts_reg_v1(TS, voxel_size):
    """
    Wrapped that performs the process of co-registering a whole time series

    """

    # Create list of ITK images
    images = []
    nint = np.size(TS, 3)
    for i in range(nint):
        images.append(create_image(TS[:, :, :, i], voxel_size))

    registrations = []
    regouts = []
    reg_df = []

    ImageType = type(images[0])
    MsMetricType = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType]

    # Register data
    for i in range(nint):
        reg, optimizer, reg_out = itk_versor3Dreg_v1(images[0], images[i],
                                                     MetricType=MsMetricType, verbose=False)
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

    result = {'TSreg': TS_out, 'Registrations': registrations,
              'Regouts': regouts, 'RegDf': reg_df}

    return result


def make_opt_par():
    """
    Recommendations and interpretation of the different parameters
    - Metric
        Default is MS (mean squares)

    - Opt range
        Expected range of motion in deg and mm

    - Learning rate
        The initial step size in the opimiser. Too large and it will be unstable.
        Too small and it might not reach the minimum

    - Relaxation factor
        The fraction by which the step size is reduced every time the optimiser
        changes direction. Too small value will reduce the step size too quickly
        and can risk local minima. Too large and the optimiser might need too many
        itterations. For noisy data the optimiser might change a lot and a higher
        value might be good.
    """
    D = {'metric': 'MS',
         'opt_range': [10, 30],
         'learning_rate': 0.2,
         'min_step_length': 0.001,
         'relax_factor': 0.6}

    return D


def merlin_ts_reg_v2(images, nint, optimizer_par):
    registrations = []
    regouts = []
    reg_df = []

    ImageType = type(images[0])
    MsMetricType = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType]

    # Register data
    print('Registering images', flush=True)
    for i in tqdm.tqdm(range(nint)):
        reg, reg_out = itk_versor3Dreg_v1(
            images[0], images[i], verbose=False, **optimizer_par)
        reg.Update()
        registrations.append(reg)
        regouts.append(reg_out)

    # Transform input data
    [nx, ny, nz] = np.shape(itk.array_from_image(images[0]))
    TS_out = np.zeros((nx, ny, nz, nint))

    for i in range(nint):
        reg_img = versor_resample(registrations[i], images[i], images[0])
        TS_out[:, :, :, i] = itk.array_from_image(reg_img)

    names = ['Int %d' % (x+1) for x in range(nint)]
    reg_df = versor_reg_summary(registrations, regouts, names, doprint=False)

    reg_res = {'Registrations': registrations,
               'Regouts': regouts, 'RegDf': reg_df, 'TS_out': TS_out}

    return reg_res


def convert_itk_matrix(m):
    np_m = np.ndarray([3, 3])
    for i in range(3):
        for j in range(3):
            np_m[i, j] = m(i, j)

    return np_m


def reg_to_kcorr(registration, traj, spacing):
    """
    Convert itk registration parameters to k-space correction

    Input:
        - registration: ITK registration object
        - traj: k-space coordinates
        - spacing: Array with spacing of data to be corrected
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

    # Rotation matrix for trajectory
    R = matrix
    traj_rot = np.matmul(traj, R)

    dx = -regParameters[3]/spacing[0]
    dy = -regParameters[4]/spacing[1]
    dz = -regParameters[5]/spacing[2]

    xF = traj[:, :, 0]/np.max(traj[:, :, 0])/2
    yF = traj[:, :, 1]/np.max(traj[:, :, 1])/2
    zF = traj[:, :, 2]/np.max(traj[:, :, 2])/2

    H = np.exp(-2j*np.pi*(xF*dx + yF*dy + zF*dz))

    return H, R, traj_rot


def calculate_TS_corrections(reg_list, raw, traj, voxel_size, nint):

    [nrcv, nspokes, npts] = np.shape(raw)
    H_all = np.zeros((nspokes, npts), dtype='complex')
    traj_all = np.zeros((nspokes, npts, 3))

    spi = int(nspokes/nint)
    for i in range(nint):
        i0 = i*spi
        i1 = (i+1)*spi
        H, R, traj_rot = reg_to_kcorr(
            reg_list[i], traj[i0:i1, :, :], 3*[voxel_size])
        H_all[i0:i1, :] = H
        traj_all[i0:i1, :, :] = traj_rot

    return H_all, traj_all


def norm_rmsd(TS, i0, i1):
    """
    Calculates the normalised root mean squared deviation between two
    images (i0, and i1) in a time series (TS). Normalisation is done
    by average of first volume.

    Input:
        - TS: Times series data (4D)
        - i0: Index volume 1
        - i1: Index volume 2

    Output:
        - NRMSD: Normalised Root Mean Squared Diffeerence

    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """

    n_vox = np.prod(np.shape(TS[:, :, :, i0]))
    SqD = (abs(TS[:, :, :, i0])-abs(TS[:, :, :, i1]))**2
    RMSD = np.sqrt(np.sum(SqD.flatten())/n_vox)
    NRMSD = RMSD/np.mean(abs(TS[:, :, :, i0]))

    return NRMSD


def normalise_timeseries(TS, iref, nlevels=1024, npoints=10):
    """
    Normalise time series data using histogram normalisation
    """

    TS_norm = np.zeros_like(TS)
    for i in range(np.shape(TS)[3]):
        TS_norm[:, :, :, i] = itk.histogram_matching_image_filter(
            abs(TS[:, :, :, i]), reference_image=abs(TS[:, :, :, iref]),
            number_of_histogram_levels=nlevels, number_of_match_points=npoints)

    return TS_norm
