import numpy as np
import copy
from sklearn.base import BaseEstimator
from scipy.ndimage import rotate


def load_images(filenames):
    '''Expects filenames to be a list of .fits file locations'''
    from astropy.io.fits import getdata

    return [ normalized_positive_image(getdata(filename)) for filename in filenames ]

def normalized_positive_image(image) :
    '''Images need to be numpy arrays between -1 and 1 for median and
    possibly HOG, but also should be log normalized so contrast is
    maintained.'''
    
    pos_def = np.clip(image,1e-6,1e100)+1.0
    return np.log(pos_def) / abs(np.log(pos_def)).max()

def rotate_images(degrees, images_X, images_y) :

    rotated_images_X = []
    rotated_y = []
    for d in degrees :
        rotated_images_X += [ rotate(image, d) for image in images_X ]
        rotated_y += images_y
    return rotated_images_X, rotated_y

class MedianSmooth(BaseEstimator):
    def __init__(self, radius = 3):
        self.radius = radius

    def fit(self, images, y = None):
        return self

    def transform(self, images):
        from skimage.filters.rank import median
        from skimage.morphology import disk
        return np.array([median(image, disk(self.radius))
                         for image in images])

    def fit_transform(self, images, y = None):
        return self.transform(images)

class Clip(BaseEstimator) :
    '''Numpy clip'''

    def __init__( self, clip_min=1e-6, clip_max=1e100):

        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, image, y=None) : 

        return self

    def transform( self, images ) :

        return np.array( [ np.clip( image, self.clip_min, self.clip_max ) 
                           for image in images ] )

    def fit_transform( self, images, y = None ) :

        return self.transform(images)

class LogPositiveDefinite(BaseEstimator) :
    ''' Shift all values to positive definite with options for taking
    log of image and making log positive definite. Return normalized
    values.'''

    def __init__( self, log = True ) : 
        self.log = log

    def fit( self, images, y = None ) :
        return self

    def _make_positive( self, image ) :
        ''' Ensure that the minimum value is just above zero. '''
        return image - image.min() + np.abs(image.min())

    def _normalize( self, image ) :
 
        if self.log : 
            return np.log( self._make_positive(image) ) / np.log( self._make_positive(image) ).max()
        else :
            return self._make_positive(image) / self._make_positive(image).max()

    def transform( self, images ) :
        return np.array( [ self._normalize(image) for image in images ] )


class HOG(BaseEstimator):
    def __init__(self, orientations = 9, pixels_per_cell = (8, 8),
                 cells_per_block = (3, 3) ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, images, y = None):
        return self

    def transform(self, images):
        from skimage.feature import hog
        return np.array([hog(image,
                             orientations = self.orientations,
                             pixels_per_cell = self.pixels_per_cell,
                             cells_per_block = self.cells_per_block,
                             )
                         for image in images])

    def fit_transform(self, images, y = None):
        return self.transform(images)
