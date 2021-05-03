import argparse

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity

"""
Tools for investigating image quality
"""


def ssim(img1, img2, kw=11, sigma=0):

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
