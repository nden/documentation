import numpy as np
from pyasdf import AsdfFile
from matplotlib import image as mplimage
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.modeling import models


def create_channel_selector(alpha, lam, channel, beta, ch_v2, ch_v3):
    if channel == 1:
        nslice = range(101, 122)#21
    elif channel == 2:
        nslice = range(201, 218) #17
    elif channel == 3:
        nslice = 16
    elif channel == 4:
        nslice = 12
    else:
        raise ValueError("Incorrect channel #")

    # transformation from local system (alpha, beta) to V2/V3
    p_v2 = models.Polynomial2D(2)
    p_v3 = models.Polynomial2D(2)
    p_v2.c0_0, p_v2.c0_1, p_v2.c1_0, p_v2.c1_1 = ch_v2[1:]
    p_v3.c0_0, p_v3.c0_1, p_v3.c1_0, p_v3.c1_1 = ch_v3[1:]
    ab_v2v3 = p_v2 & p_v3

    ind = []
    for i in range(5):
        for j in range(5):
            ind.append((i, j))

    selector = {}
    # In the paper the formula is (x-xs)^j*y^i, so the 'x' corresponds
    # to y in modeling. - swapped in Mapping
    axs = alpha.field('x_s')
    lxs = lam.field('x_s')
    #for i in range(nslice):
    for i, sl in enumerate(nslice):
        ashift = models.Shift(axs[i])
        lshift = models.Shift(lxs[i])
        palpha = models.Polynomial2D(8)
        plam = models.Polynomial2D(8)
        for index, coeff in zip(ind, alpha[i][1:]):
            setattr(palpha, 'c{0}_{1}'.format(index[0], index[1]), coeff)
        for index, coeff in zip(ind, lam[i][1:]):
            setattr(plam, 'c{0}_{1}'.format(index[0], index[1]), coeff)
        alpha_model = ashift & models.Identity(1) | palpha
        lam_model = lshift & models.Identity(1) | plam
        beta_model = models.Const1D(beta[0] + (i - 1) * beta[1])
        # Note swapping of axes
        a_b_l = models.Mapping(( 1, 0, 0, 1, 0)) | alpha_model & beta_model & lam_model
        v2_v3_l = a_b_l | models.Mapping((0, 1, 0, 1, 2)) | ab_v2v3 & models.Identity(1)
        selector[sl] = v2_v3_l
    # return alpha, beta, lambda
    return selector


def miri_models(fname):
    #f = fits.open('MIRI_FM_MIRIIFUSHORT_12_SHORT_DISTRORTION_DRAFT.fits')
    f = fits.open(fname)
    beta1 = f[0].header['B_ZERO1'], f[0].header['B_DEL1']
    beta2 = f[0].header['B_ZERO2'], f[0].header['B_DEL2']
    alpha1 = f[2].data
    lam1 = f[3].data
    alpha2 = f[5].data
    lam2 = f[5].data
    ab_v2v3 = f[6].data
    ch1_v2 = ab_v2v3[0]
    ch1_v3 = ab_v2v3[1]
    ch2_v2 = ab_v2v3[2]
    ch2_v3 = ab_v2v3[3]
    ch1_sel = create_channel_selector(alpha1, lam1, 1, beta1, ch1_v2, ch1_v3)
    ch2_sel = create_channel_selector(alpha2, lam2, 2, beta2, ch2_v2, ch2_v3)
    f.close()
    return ch1_sel, ch2_sel