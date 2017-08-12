import numpy as np
import copy
import glob

def load_images(filenames):
    '''Expects filenames to be a list of .fits file locations'''
    from astropy.io.fits import getdata
    return [ getdata(filename).copy() for filename in filenames ]
    #return [ normalized_positive_image(getdata(filename)) for filename in filenames ]

def normalized_positive_image(image) :
    '''Images need to be numpy arrays between -1 and 1 for median and
    possibly HOG, but also should be log normalized so contrast is
    maintained.'''
    
    pos_def = np.clip(image,1e-6,1e100)+1.0
    return np.log(pos_def) / abs(np.log(pos_def)).max()


def load_data(non_lens_glob, lens_glob) :
    '''
    |
    |   Load the data. 
    |   Returns X, y, filenames
    |   X is a list of numpy arrays which are the images. 
    |   y is a list of corresponding 0's and 1's
    '''

    non_lens_filenames = glob.glob(non_lens_glob)
    lens_filenames = glob.glob(lens_glob)
    filenames = non_lens_filenames + lens_filenames

    return load_images(filenames), [0] * len(non_lens_filenames) + [1] * len(lens_filenames), filenames
