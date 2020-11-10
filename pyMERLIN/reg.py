import os
import itk
import tqdm
import h5py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .dataIO import create_image

from IPython.display import display, HTML


def brain_mask(input_image, hole_radius=5, dilation=3, gauss_variance=100, gauss_max_ker=30):
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


def versor_reg_summary(registrations, reg_outs, names=None, doprint=True, show_legend=True):
    """
    Summarise results from one or more versor registration experiments

    Inputs:
        - registrations: List or registration objects
        - reg_outs: List of dictionaries of registration outputs
        - names: List of names

    Outputs:
        - df: Dataframe with summary of results
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


def versor_watcher(reg_out, optimizer, verbose):
    if verbose:
        print("{:s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s} \t {:6s}".format(
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

        # Printing
        if verbose:
            print("{:d} \t {:6.5f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f} \t {:6.3f}".format(
                cit, cv, np.rad2deg(cpos[0]), np.rad2deg(cpos[1]), np.rad2deg(cpos[2]), cpos[3], cpos[4], cpos[5]))

    return opt_watcher


def winsorize_image(image, p_low, p_high):
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


def ants_pyramid(fixed_image_fname, moving_image_fname, moco_output_name=None, fixed_output_name=None,
                 fixed_mask_fname=None, opt_range=[10, 30],
                 relax_factor=0.5, winsorize=[0.005, 0.995], verbose=True):
    """
    Perform 3D versor registration between two images

    Inputs:
        - fixed_image_fname: Fixed image (h5 file)
        - moving_image_fname: Moving image (h5 file)
        - moco_output_name: Save moco image as nifti
        - fixed_output_name: Save reference image as nifti
        - fixed_mask_fname: Mask for moving image (h5 file)
        - opt_range: Range of expected motion [deg, mm] (10, 30)
        - relax_factor: Relaxation factor for optimizer (0.5)
        - verbose: Show log output (True)
        - winsorize: Winsorize input data ([0.005, 0.995])

    Outputs:
        - registration: Registration object
        - reg_out: Registration debug stuff
    """

    # Read in data
    PixelType = itk.D
    ImageType = itk.Image[PixelType, 3]

    f_fixed = h5py.File(fixed_image_fname, 'r')
    f_move = h5py.File(moving_image_fname, 'r')

    data_fixed = f_fixed['data/0000'][:]
    data_move = f_move['data/0000'][:]

    fixed_image = create_image(
        data_fixed, f_fixed['info'][0][1], dtype=PixelType)
    moving_image = create_image(
        data_move, f_fixed['info'][0][1], dtype=PixelType)
    f_fixed.close()
    f_move.close()

    # Winsorize filter
    if winsorize:
        print("[REG] Winsorising images")
        fixed_win_filter = winsorize_image(
            fixed_image, winsorize[0], winsorize[1])
        moving_win_filter = winsorize_image(
            moving_image, winsorize[0], winsorize[1])

        fixed_image = fixed_win_filter.GetOutput()
        moving_image = moving_win_filter.GetOutput()

    # Setup image metric
    print("[REG] Setting up registration")
    metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType,
                                                             ImageType].New()
    metric.SetNumberOfHistogramBins(16)
    metric.SetUseMovingImageGradientFilter(False)
    metric.SetUseFixedImageGradientFilter(False)

    # Setup versor transform
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
    optimizer = itk.RegularStepGradientDescentOptimizerv4[PixelType].New()
    OptimizerScalesType = itk.OptimizerParameters[PixelType]
    optimizerScales = OptimizerScalesType(
        initialTransform.GetNumberOfParameters())

    # Setup registration
    registration = itk.ImageRegistrationMethodv4[ImageType,
                                                 ImageType].New()
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetInitialTransform(initialTransform)

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

    # Specifications of the Optimizer
    # Reduce later to different number per level
    optimizer.SetNumberOfIterations(500)
    optimizer.SetLearningRate(0.1)          # Default in ANTs
    optimizer.SetRelaxationFactor(relax_factor)
    optimizer.SetConvergenceWindowSize(10)
    optimizer.SetMinimumConvergenceValue(1E-6)

    # One level registration without shrinking and smoothing
    numberOfLevels = 3
    shrinkFactorsPerLevel = itk.Array[itk.F](3)
    shrinkFactorsPerLevel[0] = 4
    shrinkFactorsPerLevel[1] = 2
    shrinkFactorsPerLevel[2] = 1

    smoothingSigmasPerLevel = itk.Array[itk.F](3)
    smoothingSigmasPerLevel[0] = 2
    smoothingSigmasPerLevel[0] = 1
    smoothingSigmasPerLevel[0] = 0

    registration.SetNumberOfLevels(numberOfLevels)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel)
    registration.SetShrinkFactorsPerLevel(shrinkFactorsPerLevel)

    if fixed_mask_fname:
        CastFilterType = itk.RescaleIntensityImageFilter[MaskType, MaskType]
        mask_cast = CastFilterType.New()
        mask_cast.SetOutputMinimum(0)
        mask_cast.SetOutputMaximum(1)
        mask_cast.SetInput(mask_reader.GetOutput())
        mask_cast.Update()
        mask_out = mask_cast.GetOutput()

        MaskType = itk.ImageMaskSpatialObject[3]
        spatial_mask = MaskType.New()
        spatial_mask.SetImage(mask_out)
        spatial_mask.Update()
        # metric.SetFixedImageMask(spatial_mask)

    # Watch the itteration events
    reg_out = {'cv': [], 'tX': [], 'tY': [], 'tZ': [],
               'vX': [], 'vY': [], 'vZ': [], 'sl': [], 'lrr': []}

    print("[REG] Running Registration")
    wf = versor_watcher(reg_out, optimizer, verbose)
    optimizer.AddObserver(itk.IterationEvent(), wf)

    # --> Run registration
    registration.Update()

    # Resample moving data
    print("[REG] Resample moving image")
    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()

    TransformType = itk.VersorRigid3DTransform[itk.D]
    finalTransform = TransformType.New()
    finalTransform.SetFixedParameters(
        registration.GetOutput().Get().GetFixedParameters())
    finalTransform.SetParameters(final_parameters)

    ResampleFilterType = itk.ResampleImageFilter[ImageType,
                                                 ImageType]
    resampler = ResampleFilterType.New()
    resampler.SetTransform(finalTransform)
    resampler.SetInput(moving_image)

    resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resampler.SetOutputOrigin(fixed_image.GetOrigin())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(fixed_image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    # Write output
    if moco_output_name:
        print("[REG] Writing moco output image")
        writer = itk.ImageFileWriter[ImageType].New()
        writer.SetFileName(moco_output_name)
        writer.SetInput(resampler.GetOutput())
        writer.Update()

    if fixed_output_name:
        print("[REG] Writing reference image")
        writer = itk.ImageFileWriter[ImageType].New()
        writer.SetFileName(fixed_output_name)
        writer.SetInput(fixed_image)
        writer.Update()

    return registration, reg_out


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

    wf = versor_watcher(reg_out, optimizer, verbose)
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

    wf = versor_watcher(reg_out, optimizer, verbose)
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

    wf = versor_watcher(reg_out, optimizer, verbose)
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


def apply_moco_h5(h5_file, R, phase_corr):
    pass
