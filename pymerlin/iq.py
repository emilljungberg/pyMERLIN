import argparse

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity
from skimage.filters import sobel
from skimage.feature import canny

"""
Tools for investigating image quality
"""


def ssim(img1, img2, kw=11, sigma=0):
    """Calculate the Structural Similarity Index Measure


    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). 
    Image quality assessment: From error visibility to structural similarity. 
    IEEE Transactions on Image Processing, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861

    Args:
        img1 (np.array): Reference Image
        img2 (np.array): Comparison image
        kw (int, optional): Kernel width. Defaults to 11.
        sigma (int, optional): Gaussian window sigma. Defaults to 0.

    Returns:
        float, np.array: Mean SSIM, SSIM image
    """

    img1 /= np.quantile(img1[...], 0.99)
    img2 /= np.quantile(img2[...], 0.99)

    gauss_window = False
    if sigma:
        gauss_window = True

    mssim, S = structural_similarity(
        img1, img2, win_size=kw, data_range=1, gradient=False,
        multichannel=False, gaussian_weights=gauss_window, full=True, use_sample_covariance=True,
        sigma=sigma)

    return mssim, S


def aes(img, mask=None, canny_edges=None, canny_sigma=2):
    """Calculate the Average Edge Strength

    Reference:
    Aksoy, M., Forman, C., Straka, M., Çukur, T., Hornegger, J., & Bammer, R. (2012). 
    Hybrid prospective and retrospective head motion correction to mitigate cross-calibration errors. 
    Magnetic Resonance in Medicine, 67(5), 1237–1251. https://doi.org/10.1002/mrm.23101


    Args:
        img (np.array): Image
        mask (np.array, optional): Brain mask. Defaults to None.
        canny_edges (np.array, optional): Edges to use for calculation, calculates if `None`. Defaults to None.
        canny_sigma (int, optional): Sigma for canny edge detection filter. Defaults to 2.

    Returns:
        float, np.array, np.array: aes, edges, canny edge mask
    """

    imax = np.quantile(img[mask == 1], 0.99)

    if canny_edges is None:
        canny_edges = np.zeros_like(img)
        for z in range(img.shape[2]):
            canny_edges[:, :, z] = canny(
                img[:, :, z], sigma=canny_sigma)

    canny_edges *= mask

    img_edges = sobel(img/imax) * canny_edges
    aes = np.mean(img_edges[canny_edges == 1])

    return aes, img_edges, canny_edges


def nrmse(img_ref, img_comp, mask):
    """Calculates the Normalised root mean squared error

    Args:
        img_ref (np.array): Reference image
        img_comp (np.array): Comparison image
        mask (np.array): Brain mask

    Returns:
        np.array: nrmse
    """

    yrange = np.max(img_ref[mask == 1]) - np.min(img_ref[mask == 1])

    return np.sqrt(np.sum((img_comp[mask == 1] - img_ref[mask == 1])**2) / np.sum(mask)) / yrange


def gradient_entropy(img):
    """Calculates gradient entropy of image

    Reference:
    McGee, K.P., Manduca, A., Felmlee, J.P., Riederer, S.J. and Ehman, R.L. (2000), 
    Image metric‐based correction (Autocorrection) of motion effects: Analysis of image metrics. 
    J. Magn. Reson. Imaging, 11: 174-181. 
    https://doi.org/10.1002/(SICI)1522-2586(200002)11:2<174::AID-JMRI15>3.0.CO;2-3


    Args:
        img (np.array): Image

    Returns:
        float: gradient entropy
    """

    """
    Calculates gradient entropy of ND image

        """

    h = img/np.sum(img**2)
    GE = -np.sum(h*np.log2(h))
    return GE
