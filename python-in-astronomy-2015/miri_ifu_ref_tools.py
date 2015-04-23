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


def fits_column_to_model(fits_column, x_degree, y_degree):
    """
    Create a modeling.Model instance from an array of coefficients
    read in from MIRI d2c and c2d files.

    Parameters
    ----------
    fits_column : nd.array
        array with coefficients from column in binary table
    x_degree : int
        order of polynomial in x
    y_degree : int
        order of polynomial in y
    """
    n_coeff = len(fits_column)
    ind = []
    for i in range(x_degree+1):
        for j in range(y_degree+1):
            ind.append((i, j))
    model = models.Polynomial2D(degree=x_degree+y_degree)
    for i,j in zip(ind, fits_column):
        coeff_name = 'c' + str(i[0]) + '_' + str(i[1])
        setattr(model, coeff_name, j)
    return model


def miri_old_model():
    d2c=fits.open('polyd2c_1A_01.00.00.fits')
    slices = d2c[1].data
    rids = np.unique(slices).tolist()
    rids.remove(0)

    shiftx = models.Shift(-d2c[0].header['CENTERX'])
    shifty = models.Shift(-d2c[0].header['CENTERY'])
    scalex = models.Scale(1. / d2c[0].header['NORMX'])
    scaley = models.Scale(1. / d2c[0].header['NORMY'])

    b1 = d2c[1].header['B_DEL']
    b0 = d2c[1].header['B_MIN']

    coeffs = d2c[2].data
    channel_selector = {}
    for i in rids:
        palpha = fits_column_to_model(coeffs.field("alpha_"+str(int(i))), 4, 4)
        plam = fits_column_to_model(coeffs.field("lambda_"+str(int(i))), 4, 4)
        malpha = (shiftx & shifty) | (scalex & scaley) | palpha
        mlam = (shiftx & shifty) | (scalex & scaley) | plam

        beta = models.Const1D(b0 + i*b1)

        channel_selector[i] = models.Mapping((0, 1, 0, 0, 1)) | malpha & beta & mlam
    return channel_selector

def rs():
    from gwcs import selector
    ch1_sel=miri_old_model()
    fd2c=fits.open('polyd2c_1A_01.00.00.fits')
    slices=fd2c[1].data
    miri_mask=selector.SelectorMask(slices)
    rs=selector.RegionsSelector(inputs=['x','y'], outputs=['v2', 'v3', 'lambda'], selector=ch1_sel, selector_mask=miri_mask)
    x, y= np.mgrid[:1024, :1032]
    #ra, dec, lam=rs(x, y)
    return rs

"""
alpha - models.Chebyshev2D(x_degree=2, y_degree=1)

lam = models.Chebbyshev2D(x_degree=1, y_degree=1)
lam.c0_1.fixed=True
lam.c1_1.fixed=True


"""
from astropy.modeling import models, fitting
import copy
def make_channel_models(channel):
    f = fits.open('MIRI_FM_LW_A_D2C_01.00.00.fits')
    if channel == 3:
        slice_mask = f[3].data[:,:500]
        b1 = f[0].header['B_DEL3']
        b0 = f[0].header['B_MIN3']

    elif channel == 4:
        slice_mask = f[3].data[:, 500:]
        b1 = f[0].header['B_DEL4']
        b0 = f[0].header['B_MIN4']

    slices = np.unique(slice_mask)
    slices = np.asarray(slices, dtype=np.int16).tolist()
    slices.remove(0)
    
    #read alpha and lambda planes from pixel maps file
    lam = f[1].data
    alpha = f[2].data
    # create a model to fit to each slice in alpha plane
    amodel = models.Chebyshev2D(x_degree=2, y_degree=1)
    # a model to be fitted to each slice in lambda plane
    lmodel = models.Chebyshev2D(x_degree=1, y_degree=1)
    lmodel.c0_1.fixed = True
    lmodel.c1_1.fixed = True
    fitter = fitting.LinearLSQFitter()
    reg_models = {}
    for sl in slices:
        #print('sl', sl)
        ind = (slice_mask==sl).nonzero()#(slice_mask[:, :500] ==sl).nonzero()
        x0 = ind[0].min()
        x1 = ind[0].max()
        y0 = ind[1].min()
        y1 = ind[1].max()
        if channel ==4:
            y0 += 500
            y1 += 500
        x, y = np.mgrid[x0:x1, y0:y1]
        sllam = lam[x0:x1, y0:y1]
        slalpha = alpha[x0:x1, y0:y1]
        lfitted = fitter(lmodel, x, y, sllam)
        afitted = fitter(amodel, x, y, slalpha)

        if channel == 4:
            beta_model = models.Const1D(b0 + b1 * (sl+12))
            reg_models[sl+12] = lfitted, afitted, beta_model, (x0, x1, y0, y1)
        else:
            beta_model = models.Const1D(b0 + b1*sl)
            reg_models[sl] = lfitted, afitted, beta_model, (x0, x1, y0, y1)

    return reg_models

def plot_alpha(reg_models):
    nalpha = np.zeros((1025, 1033))
    nlam = np.zeros((1025, 1033))
    for i in range(1, 29):
        t = reg_models[i]
        x0, x1, y0, y1 = t[3]
        x, y = np.mgrid[x0:x1, y0:y1]
        alpha = t[1](x, y)
        nalpha[x, y] = alpha
        lam = t[0](x, y)
        nlam[x, y] = lam
    return nalpha, nlam

def make_models():
    reg_models3 = make_channel_models(3)
    reg_models4 = make_channel_models(4)
    reg_models = copy.deepcopy(reg_models3)
    reg_models.update(reg_models4)
    return reg_models