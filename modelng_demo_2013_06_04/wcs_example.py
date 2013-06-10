from modeling import SCompositeModel, LabeledInput
from modeling import models, projections


class Pix2Sky(SCompositeModel):
    """
    An example of pixel to sky transformation without distortion

    Parameters
    ----------

    crpix: a list or tuple of 2 floats
        CRPIX values
    pc: numpy array of shape (2,2)
        PC or CD matrix
    phip, thetap, lonp : float
        Euler angles in degrees
    projcode: string
        One of projection codes in projections.projcode
    projpars: dict
        projection parameters (if any)

    """
    def __init__(self, crpix, pc, phip, thetap, lonp, projcode, projparams={}):

        self.projcode = projcode.upper()
        self.crpix = crpix
        self.pc = pc
        self.phip = phip
        self.lonp = lonp
        self.thetap = thetap
        if projcode not in projections.projcodes:
            raise ValueError('Projection code %s, not recognized' %projcode)

        # create a projection transform
        projklassname = 'Pix2Sky_' + self.projcode
        projklass = getattr(projections, projklassname)
        prj = projklass(**projparams)

        # Create a RotateNative2Celestial transform using the Euler angles
        n2c = models.RotateNative2Celestial(phip, thetap, lonp)

        # create shift transforms in x and y
        offx = models.ShiftModel(-crpix[0])
        offy = models.ShiftModel(-crpix[1])

        # create a CD/PC transform
        pcrot = models.MatrixRotation2D(rotmat=pc)

        # finally create the composite transform which does pixel to sky transformation
        SCompositeModel.__init__(self, [offx, offy, pcrot, prj, n2c],
                                 inmap=[['x'], ['y'], ['x', 'y'], ['x', 'y'], ['x', 'y']],
                                 outmap=[['x'], ['y'], ['x', 'y'],['x', 'y'], ['x','y']])
        self.has_inverse = True

    def inverse(self, projparams={}, phip=0, thetap=-90, lonp=180):
        return Sky2Pix(self.crpix, self.pc, self.phip, self.thetap, self.lonp, self.projcode)

    def __call__(self, x, y):
        # create a LabeledObject from x and y
        labeled_input = LabeledInput([x,y], ['x', 'y'])
        # pass the LabeledObject to the composite model to do the transformation
        result = SCompositeModel.__call__(self, labeled_input)
        return result.x, result.y

class Sky2Pix(SCompositeModel):
    """
    An example of sky to pixel transformation without distortion

    Parameters
    ----------
    crpix: a list or tuple of 2 floats
              CRPIX values
    pc: numpy array of shape (2,2)
              PC or CD matrix
    phip, thetap, lonp - float
              Euler angles in degrees
    projcode: string
             One of projection codes in projections.projcode
    projpars: dict
             projection parameters (if any)

    """
    def __init__(self, crpix, pc, phip, thetap, lonp, projcode, projparams={}):

        self.projcode = projcode.upper()
        self.crpix = crpix
        self.pc = pc
        self.thetap = thetap
        self.lonp = lonp
        self.phip = phip

        if projcode not in projections.projcodes:
            raise ValueError('Projection code %s, not recognized' %projcode)

        # create a projectiontransform
        projklassname = 'Sky2Pix_' + projcode
        projklass = getattr(projections, projklassname)
        prj = projklass(**projparams)

        # determine the Euler angles and create a RotateCelestial2Native transform
        c2n = models.RotateCelestial2Native(phip, thetap, lonp)

        # create shift transforms in x and y
        offx = models.ShiftModel(crpix[0])
        offy = models.ShiftModel(crpix[1])

        # create a transform for the CD/PC matrix
        pcrot = models.MatrixRotation2D(rotmat=pc).inverse()

        # create the composite transform
        SCompositeModel.__init__(self, [c2n, prj, pcrot, offx, offy],
                                 inmap=[['x', 'y'], ['x', 'y'], ['x', 'y'], ['x'], ['y']],
                                 outmap=[['x', 'y'], ['x', 'y'], ['x', 'y'], ['x'], ['y']])

    def inverse(self):
        return Pix2Sky(self.crpix, self.pc, self.phip, self.thetap, self.lonp, self.projcode)

    def __call__(self, alpha, delta):
        # create a LabeledObject form the input x and y coordinates
        labeled_input = LabeledInput([alpha, delta], ['x', 'y'])

        # pass the LabeledInput object to the composite transform
        result = SCompositeModel.__call__(self, labeled_input)
        return result.x, result.y

