'''
Tools necessary to save a trained model, load a trained model, and to
make predictions from a trained model on a test set and collect false
predictions.
'''

import numpy as np
import glob
import cPickle as pickle
import time
from StrongCNN.IO.load_images import load_data
from StrongCNN.utils.features_selection import get_false_predictions_list, get_filenames_in_threshold_range
from StrongCNN.utils.model_info import get_scores

def train_model(model_pipeline, X_train, y_train, **params) :
    '''
    | Train the model on given training set and parameters
    '''
    
    # Create the grid search with list of parameters
    # to search.  All values are now tuples
    model_pipeline.set_params(**params).fit(X_train, y_train)


def dump_model(pklfile, trained_model) :
    '''Save a trained model'''

    with open(pklfile,'wb') as output :
        pickle.dump(trained_model, output, -1)
    

def load_model(pklfile) :
    '''Returns a trained model that can make predictions'''

    with open(pklfile,'r') as output :
        return pickle.load(output)


def generate_X_y(non_lens_glob, lens_glob) :
    '''Reads in data that will be features and targets, outputs as numpy array data'''
    import glob
    filenames = glob.glob(non_lens_glob)+glob.glob(lens_glob)
    X, y = load_data(non_lens_glob, lens_glob)

    return X, y, filenames

def generate_X_scores( model, X, y, filenames ) :
    '''Reads in features, prints filenames and associated scores '''
    return filenames, get_scores( model, X ), y

