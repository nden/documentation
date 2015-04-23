import numpy as np
from pyasdf import AsdfFile
from matplotlib import image as mplimage
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.modeling import models
from gwcs import wcs, selector
import copy

import miri_ifu_ref_tools

def read_model(filename):
    f = AsdfFile.read(filename)
    return f.tree['model']


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """
    import matplotlib
    from matplotlib import pyplot as plt
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


def show(mask, n_regions=3):
    from matplotlib import pyplot as plt
    c = cmap_discretize('jet', n_regions)
    labels=np.arange(n_regions)
    loc= labels
    f = plt.figure()
    #f.set_size_inches(12, 8, forward=True)
    plt.imshow(mask, interpolation='nearest', cmap=c, aspect='auto')
    cb = plt.colorbar()
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    plt.show()

#def display_regions_image():
    #img = mplimage.imread('ifu-regions.png')
    #plt.imshow(img[::-1])


def read_regions():
    data = fits.getdata('MIRI_FM_MIRIIFUSHORT_12_SHORT_DISTRORTION_DRAFT.fits')
    data = data[:, :500]
    for i in range(1,12):
        data[data==100+i] = i
    for i in range(12,22):
        data[data==100+i] = 0
    #show(data, 11)
    return data


def create_channel_selector(alpha, lam, channel, beta):
    if channel == 1:
        nslice = 21
    elif channel == 2:
        nslice = 17
    elif channel == 3:
        nslice = 16
    elif channel == 4:
        nslice = 12
    else:
        raise ValueError("Incorrect channel #")
    ind = []
    for i in range(5):
        for j in range(5):
            ind.append((i, j))

    selector = {}
    # In the paper the formula is (x-xs)^j*y^i, so the 'x' corresponds
    # to y in modeling. - swapped in Mapping
    axs = alpha.field('x_s')
    lxs = lam.field('x_s')
    for i in range(nslice):
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
        selector[i] = models.Mapping((1, 0, 1, 0, 0)) | alpha_model & lam_model & beta_model

    return selector


# def miri_models(fname):
#     #f = fits.open('MIRI_FM_MIRIIFUSHORT_12_SHORT_DISTRORTION_DRAFT.fits')
#     f = fits.open(fname)
#     beta1 = f[0].header['B_ZERO1'], f[0].header['B_DEL1']
#     beta2 = f[0].header['B_ZERO2'], f[0].header['B_DEL2']
#     alpha1 = f[2].data
#     lam1 = f[3].data
#     alpha2 = f[5].data
#     lam2 = f[5].data
#     ch1_sel = create_channel_selector(alpha1, lam1, 1, beta1)
#     ch2_sel = create_channel_selector(alpha2, lam2, 2, beta2)
#     f.close()
#     return ch1_sel, ch2_sel

#miri-example
# def miri_models():
#     f = fits.open('MIRI_FM_MIRIIFUSHORT_12_SHORT_DISTRORTION_DRAFT.fits')
#     slices = f[1].data
#     #mask=selector.SelectorMask(slices)
#     #x,y=np.mgrid[:1024, :1024]
#     ch1, ch2=miri_ifu_ref_tools.miri_models('MIRI_FM_MIRIIFUSHORT_12_SHORT_DISTRORTION_DRAFT.fits')
#     import copy
#     sel = copy.deepcopy(ch1)
#     sel.update(ch2)
#     f.close()
#     sel1 = {}
#     for i in sel:
#         if i > 100 and i < 200:
#             sel1[i-100] = sel[i]
#         elif i >200:
#             sel1[i-200+21] = sel[i]

#     return sel1    

# def miri_mask():
#     f = fits.open('MIRI_FM_MIRIIFUSHORT_12_SHORT_DISTRORTION_DRAFT.fits')
#     slices = f[1].data
#     sl1 = np.where(slices > 100, slices-100, slices)
#     sl2 = np.where(sl1>100, sl1-100+21, sl1)
#     #mask = selector.SelectorMask(sl2)
#     return sl2

def miri_mask():
    f = fits.open('MIRI_FM_LW_A_D2C_01.00.00.fits')
    slices = f[3].data
    nslices = np.zeros((slices.shape))
    nslices[:, :500] = slices[:, :500]
    sl1 = slices[:, 500:]
    sl2 = np.where(sl1> 0, sl1+12, sl1)
    #sl2 = np.where(sl1>100, sl1-100+21, sl1)
    nslices[:,500:] = sl2
    return nslices


def miri_models_test():
    reg_models3 = miri_ifu_ref_tools.make_channel_models(3)
    reg_models4 = miri_ifu_ref_tools.make_channel_models(4)
    reg_models = copy.deepcopy(reg_models3)
    reg_models.update(reg_models4)
    return reg_models

def miri_models():
    reg_models3 = miri_ifu_ref_tools.make_channel_models(3)
    reg_models4 = miri_ifu_ref_tools.make_channel_models(4)
    reg_models = {}

    for reg in reg_models3:
        lam_model, alpha_model, beta_model, _ = reg_models3[reg]
        model = models.Mapping((0, 1, 0, 0, 1)) | alpha_model & beta_model & lam_model
        reg_models[reg] = model
    for reg in reg_models4:
        lam_model, alpha_model, beta_model, _ = reg_models4[reg]
        model = models.Mapping((0, 1, 0, 0, 1)) | alpha_model & beta_model & lam_model
        reg_models[reg] = model
    return reg_models


def create_asdf_ref_files(fname):
    #f = fits.open('../j94f05bgq_flt.fits')
    f = fits.open(fname)
    from gwcs import util
    whdr = util.read_wcs_from_header(f[1].header)
    crpix = whdr['CRPIX']
    shift = models.Shift(crpix[0]) & models.Shift(crpix[1])
    rotation = models.AffineTransformation2D(matrix=whdr['PC'])
    cdelt = whdr['CDELT']
    scale = models.Scale(cdelt[0]) & models.Scale(cdelt[1])
    #print util.get_projcode(whdr['CTYPE'][0])
    tan = models.Pix2Sky_TAN()
    crval = whdr['CRVAL']
    n2c = models.RotateNative2Celestial(crval[0], crval[1], 180)
    foc2sky = (shift | rotation | scale | tan | n2c).rename('foc2sky')
    fasdf = AsdfFile()
    fasdf.tree = {'model': foc2sky}
    fasdf.write_to('foc2sky.asdf')

