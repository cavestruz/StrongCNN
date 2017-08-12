'''
This generates one trained model, predicts on some test data, and marks the false positives/false negatives.
'''

import numpy as np
import glob
import cPickle as pickle
import time
import image_processing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from collections import Counter

def confusion_matrix(predicted, actual):
    '''
    | Outputs (what model thinks it is, what it is)

    '''
    predicted, actual = map(np.array, [predicted, actual])
    return Counter(zip(predicted, actual))

def _get_params(median=False, median_smooth__radius=3, hog__orientations=9, 
                hog__pixels_per_cell=(16,16), hog__cells_per_block=(1,1), 
                logistic_regression__C=10.) :
    return locals()


def train_model(kw) :
    '''
    | Create a pickle file of the trained model
    | returns successful predictions of the test data
    '''
    
    # Turn the tuple into a packed dictionary to get all parameters
    params = _get_params(logistic_regression__C=kw) 

    # Create the pipeline which consists of image
    # processing and a classifier
    # Note - can make this map to a dictionary of image processors instead of just median
    image_processors = [ ('hog', image_processing.HOG()) ]

    if params.pop('median') :
        image_processors.insert(0,('median_smooth', image_processing.MedianSmooth()))
    else :
        params.pop('median_smooth__radius')

    classifier = ('logistic_regression', LogisticRegression())

    estimators = image_processors + [classifier]
    
    pipeline = Pipeline(estimators)

    # Create the grid search with list of parameters
    # to search.  All values are now tuples
    pipeline.set_params(**params).fit(X_train, y_train)

    with open('pickled_models/C10_4x4_3x3.pkl','wb') as output :
        pickle.dump(pipeline, output, -1)

    # Confusion matrix is                                                                                                                            
    print "Confusion matrix on training set"
    print confusion_matrix(pipeline.predict(X_train), y_train)
    print
    print "Score on training set =", pipeline.score(X_train, y_train)

def load_model(pklfile) :
    '''Returns a trained model that can make predictions'''

    with open(pklfile,'r') as output :
        return pickle.load(output)

def get_false_predictions_list( trained_model, X, y, filenames ) :

    # Score the test set                                                                
    print "Confusion matrix on test set"
    print confusion_matrix(trained_model.predict(X), y)
    print
    print "Score on test set =", trained_model.score(X, y)


    successful_predictions = map( lambda x: x[0] == x[1], 
                                  zip( trained_model.predict( X ), y) )
    
    return [ sf[1] for sf in zip( successful_predictions, filenames ) if sf[0] == 0 ]

def generate_X_y(non_lens_glob, lens_glob) :
    '''Reads in data that will be features and targets, outputs as numpy array data'''
    non_lens_filenames = glob.glob(non_lens_glob)
    lens_filenames = glob.glob(lens_glob)
    filenames = non_lens_filenames + lens_filenames
    X = image_processing.load_images(filenames)
    y = [0] * len(non_lens_filenames) + [1] * len(lens_filenames)
    
    return X, y, filenames
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('non_lens_glob')
    parser.add_argument('lens_glob')
    parser.add_argument('non_lens_test_glob')
    parser.add_argument('lens_test_glob')
    parser.add_argument('model_pkl_name')
    parser.add_argument('train_model_bool')

    args = vars(parser.parse_args())

    C_val = 10.
    rotation_degrees = [ 0, 90, 180, 270 ]
    
    # Load the data. X is a list of numpy arrays
    # which are the images.
    X_train, y_train, filenames_train = generate_X_y( args['non_lens_glob'], args['lens_glob'] )
    X_test, y_test, filenames_test = generate_X_y( args['non_lens_test_glob'], args['lens_test_glob'] )


    # Train/test split
    X_train, y_train = image_processing.rotate_images( rotation_degrees, X_train, y_train )
    if args['train_model_bool'] :
        train_model(C_val)

    print "False Predictions: "
    print get_false_predictions_list( load_model(args['model_pkl_name']), 
                                      X_test, y_test, filenames_test )

    start_time = time.time()
    time_taken = time.time() - start_time
    print "Time to generate object:", time_taken

