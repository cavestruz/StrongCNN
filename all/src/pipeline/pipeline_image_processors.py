import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, Imputer, Normalizer
from skimage.feature import hog
from numpy.linalg import norm


class MaskAverageImpute() :
    '''
    This is used for observed data with masked bright point sources.
    It uses image_mask_avg_impute function to fill out masked values.
    '''
    def __init__( self, mask_value = 100., max_iter = 5 ) :
        self.mask_value = mask_value
        self.max_iter = max_iter

    def fit( self, images, y = None ) :
        return self

    def transform( self, images ) :
        return np.array( [ image_mask_avg_impute(image) for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform( images )


class MedianSmooth(BaseEstimator):
    '''
    This was used during testing of the HOG feature extraction in hopes of reducing effects from noise.  However, a median filter creates spurious features in the HOG visualization that worsened model performance.  HOG already deals with small scale noise well.
    '''
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

class Flatten(BaseEstimator) :
    def __init__( self, axis=None ) :
        self.axis = axis

    def fit( self, images, y = None ) :
        return self

    def transform( self, images ) :
        return np.array( [ image.flatten() for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform( images )

class UnFlatten(BaseEstimator) :
    def __init__( self, shape = (101, 101) ) :
        self.shape = shape

    def fit( self, images, y = None ) :
        return self

    def transform( self, images ) :
        return np.array( [ np.reshape( image, self.shape ) for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform( images )


class Norm(BaseEstimator) :
    def __init__( self, axis=None ) :
        self.axis = axis

    def fit( self, images, y = None ) :
        return self

    def transform( self, images ) :
        from numpy.linalg import norm
        return np.array( [ image/norm(image) for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform( images )


class Clip(BaseEstimator) :
    '''Numpy clip'''

    def __init__( self, lower=1e-6, upper=1e100):

        self.clip_min = lower
        self.clip_max = upper

    def fit(self, image, y = None) :

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

    def __init__( self, log = True, shift = None ) :
        self.log = log
        self.shift = shift

    def fit( self, images, y = None ) :
        return self

    def _make_positive( self, image ) :
        ''' Ensure that the minimum value is just above zero. '''
        if self.shift == None :
            return image - image.min() + np.abs(image.min())
        else :
            return image + self.shift

    def _normalize( self, image ) :
        if self.log :
            return np.log( self._make_positive(image) ) / np.log( self._make_positive(image) ).max()
        else :
            return self._make_positive(image) / self._make_positive(image).max()

    def transform( self, images ) :
        return np.array( [ self._normalize(image) for image in images ] )

class PreprocessHST( BaseEstimator ) :
    '''This is significantly the best set of preprocessing I've found
    for our HST like data.  This combines the Clip, Norm, and LogPositiveDefinite pipeline objects.'''
    def __init__( self, lower_clip = 1e-6, upper_clip = 1e100, shift = 1.0 ):
        self.lower_clip = lower_clip
        self.upper_clip = upper_clip
        self.shift = shift

    def fit( self, images = [], y=None ) :
        return self

    def _pos_def( self, image ) :
        return np.clip(image, self.lower_clip, self.upper_clip) + self.shift

    def transform( self, images ) :
        return np.array( [ np.log(self._pos_def(image)) / abs(np.log(self._pos_def(image))).max() for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform( images )

class MidpointSigmaClip(BaseEstimator) :
    '''
    This is useful when the noise level is not a priori known.

    Clips the data at n sigma from a midpoint.
    kwargs
    ---------
    sigma_factor : -1 means 1 sigma below the midpoint
    mid_point : option for calculating midpoint, e.g. 'median', 'mean'
    '''
    def __init__( self, sigma_factor=-1., mid_point='mean', upper_clip=1e100, normalize=0. ):
        self.sigma_factor = sigma_factor
        self.upper_clip = upper_clip
        if normalize == 1. : self.normalize = True
        else : self.normalize = False
        mid_point_options = ['median', 'mean' ]

        assert( mid_point in mid_point_options )
        self.mid_point = mid_point

    def fit( self, images, y = None ) :
        return self

    def _calc_lower_clip( self, image ) :
        if self.mid_point == 'median' :
            return np.median( image ) + image.std() * self.sigma_factor
        if self.mid_point == 'mean' :
            return np.mean( image ) + image.std() * self.sigma_factor

    def _clip( self, image ) :
        return np.clip(image, self._calc_lower_clip(image), self.upper_clip)

    def transform( self, images ) :
        if self.normalize :
            #from numpy.linalg import norm
            from sklearn.preprocessing import normalize
            return np.array([ self._clip(normalize(image)) for image in images])
        else :
            return np.array([ self._clip(image) for image in images])

    def fit_transform( self, images, y = None ) :
        return self.transform( images )

class SKPreProcessNormalize(BaseEstimator) :
    def __init__( self, normalize = True ) :
        self.normalize = normalize

    def fit( self, images, y = None ) :
        return self

    def transform( self, images ) :
        from sklearn.preprocessing import normalize
        return np.array( [ normalize(image) for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform( images )

class SKPreProcessNormalizeConcatenated(BaseEstimator) :
    '''For multi-page images'''
    def __init__( self, normalize = True ) :
        self.normalize = normalize

    def fit( self, images, y = None ) :
        return self

    def _normalize_each_page( self, image ) :
        from sklearn.preprocessing import normalize
        return np.array( [ normalize( image[i] )
                           for i in range( image.shape[0] ) ] )


    def transform( self, images ) :
        return np.array( [ self._normalize_each_page(image) for image in images ] )

    def fit_transform( self, images, y = None ) :
        return self.transform( images )


class ConcatenatedHOG(BaseEstimator) :
    '''For multiband training - specific to KiDS four band.
    Requires tuples of the hog parameterization for each band'''
    def __init__( self, orientations=(4, 5, 6, 4),
                  pixels_per_cell=((16,16),(16,16),(16,16),(16,16)),
                  cells_per_block=((3,3),(3,3),(3,3),(3,3),),
                  avg_mask=False,
                  strips=False
                  ) :
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.avg_mask=avg_mask
        self.strips=strips

    def fit( self, images, y = None ) :
        return self

    def _clip( self, image ) :
        return np.clip( image, image.mean() - image.std(), 1e100 )

    def _norm( self, image ) :
        return image / norm( image )

    def _log_pos_def( self, image ) :
        return np.log( image + 1.0 )

    def _div_by_max( self, image ) :
        return image / abs(image).max()

    def _preprocess( self, image ) :
        if self.strips :
            return np.array( [ self._div_by_max( self._log_pos_def( self._norm( self._clip( image[i] ) ) ) )
                               for i in range( image.shape[0] ) ] )
        else :
            return self._div_by_max( self._log_pos_def( self._norm( self._clip( image ) ) ) )

    def _concatenated_hog( self, image ) :
        for kwarg in [ self.orientations, self.pixels_per_cell,
                       self.cells_per_block ] :
            try :
                assert( image.shape[0] == len(kwarg) )
            except AssertionError :
                print "assertion error", image.shape[0],'!=',len(kwarg),' for ', kwarg

        if self.avg_mask :
            if len(image.shape) == 2 :
                i = 0
                return hog( self._preprocess( image_mask_avg_impute(image) ),
                                          orientations = self.orientations[i],
                                          pixels_per_cell = self.pixels_per_cell[i],
                                          cells_per_block = self.cells_per_block[i]
                                          )
            else :
                return np.concatenate( [ hog( self._preprocess( image_mask_avg_impute(image[i]) ),
                                              orientations = self.orientations[i],
                                              pixels_per_cell = self.pixels_per_cell[i],
                                              cells_per_block = self.cells_per_block[i]
                                              ) for i in range( image.shape[0] ) ] )
        else :
            if len(image.shape) == 2 :
                i = 0
                return hog( self._preprocess(image),
                            orientations = self.orientations[i],
                            pixels_per_cell = self.pixels_per_cell[i],
                            cells_per_block = self.cells_per_block[i]
                            )

            else :
                return np.concatenate( [ hog( self._preprocess(image[i]),
                                              orientations = self.orientations[i],
                                              pixels_per_cell = self.pixels_per_cell[i],
                                              cells_per_block = self.cells_per_block[i]
                                              ) for i in range( image.shape[0] ) ] )


    def transform( self, images ) :
        return np.array([ self._concatenated_hog( image ) for image in images ])


    def fit_transform(self, images, y = None):
        return self.transform(images)


class HOG(BaseEstimator):
    '''
    Primary feature extraction method tested.
    '''
    def __init__( self, orientations = 9, pixels_per_cell = (8, 8),
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


def image_mask_avg_impute(image, mask_value = 100., max_iter = 5):
    """
    Imputes pixels with mask_value with the average value of
    their neighbors. In case a pixel has all masked neighbors,
    the operation is performed iteratively until all pixels
    are imputed, or max_iter is reached.
    """
    masked_indices = zip(*np.where(image == mask_value))
    if not masked_indices:
        return image
    image = np.array(image)
    try :
        assert( len(image.shape) == 2 )
    except AssertionError :
        print image.shape, image

    masked_values = [image[i, j] for i, j in masked_indices]

    for _ in range(max_iter):
        for i, j in masked_indices:
            unmasked_neighbor_vals = [x for x in _get_neighbors(image, i, j)
                                      if x != mask_value]
            if not unmasked_neighbor_vals:
                continue
            image[i,j] = np.average(unmasked_neighbor_vals)

        masked_values_prev = masked_values
        masked_values = [image[i, j] for i, j in masked_indices]
        if masked_values_prev == masked_values:
            break

    return image

def _get_neighbors(image, i, j):
    """
    Get the values of the immediate neighbors in
    two dimensions.
    """
    neighbor_vals = []
    if i > 0:
        neighbor_vals.append(image[i-1, j])
    if i < image.shape[0] - 1:
        neighbor_vals.append(image[i+1, j])
    if j > 0:
        neighbor_vals.append(image[i, j-1])
    if j < image.shape[1] - 1:
        neighbor_vals.append(image[i, j+1])

    return neighbor_vals

image_processors = { 'median_filter' : MedianSmooth(),
                     'hog' : HOG(), # Most used
                     'clip' : Clip(),
                     'log_positive_definite' : LogPositiveDefinite(),
                     'scale' : StandardScaler(),
                     'imputer' : Imputer(),
                     'hst' : PreprocessHST(), # Most used for just HST
                     'midpointsigmaclip' : MidpointSigmaClip(),
                     'norm' : Norm(),
                     'normalizer' : Normalizer(),
                     'concatenated_hog' : ConcatenatedHOG(),
                     'flatten' : Flatten(),
                     'unflatten' : UnFlatten(),
                     'mask_avg_impute' : MaskAverageImpute() ,
                     'sknormalize' : SKPreProcessNormalize(),
                     'sknormalize_cat' :  SKPreProcessNormalizeConcatenated(),
                     }
