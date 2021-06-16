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
    """
    Structural Similarity Index Measure

    Reference:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). 
    Image quality assessment: From error visibility to structural similarity. 
    IEEE Transactions on Image Processing, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
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
    """
    Calculate the Average Edge Strength (AES)

    Reference:
    Aksoy, M., Forman, C., Straka, M., Çukur, T., Hornegger, J., & Bammer, R. (2012). 
    Hybrid prospective and retrospective head motion correction to mitigate cross-calibration errors. 
    Magnetic Resonance in Medicine, 67(5), 1237–1251. https://doi.org/10.1002/mrm.23101
    """

    imax = np.quantile(img[mask == 1], 0.99)

    if canny_edges is None:
        canny_edges = np.zeros_like(img)
        for x in range(img.shape[0]):
            canny_edges[x, :, :] = canny(img[x, :, :], sigma=canny_sigma)

    img_edges = sobel(img/imax) * canny_edges

    aes = np.mean(img_edges[canny_edges == 1])

    return aes, img_edges, canny_edges


def nrmse(img_ref, img_comp, mask):
    """
    Calculate the normalise root mean squared error 
    using max-min normalisation
    """

    yrange = np.max(img_ref[mask == 1]) - np.min(img_ref[mask == 1])

    return np.sqrt(np.sum((img_comp[mask == 1] - img_ref[mask == 1])**2) / np.sum(mask)) / yrange
