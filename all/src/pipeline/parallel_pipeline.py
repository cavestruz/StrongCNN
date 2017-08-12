import numpy as np
import glob
import time
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import image_processing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter
from multiprocessing import Pool

def confusion_matrix(predicted, actual):
    '''
    | Outputs (what model thinks it is, what it is)

    '''
    predicted, actual = map(np.array, [predicted, actual])
    return Counter(zip(predicted, actual))

def _get_params(median=False, median_smooth__radius=3, hog__orientations=9, 
                hog__pixels_per_cell=(8,8), hog__cells_per_block=(1,1), 
                logistic_regression__C=0.1) :
    return locals()


def score_for_params(kw) :
    '''
    | Get score from just one set of parameters
    | Takes in keyword arguments, including whether or not median filter will be included.
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

    pipeline.predict(X_test)
    
    return pipeline.score(X_train,y_train), pipeline.score(X_test, y_test)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('non_lens_glob')
    parser.add_argument('lens_glob')

    args = vars(parser.parse_args())

    C_vals = 10**np.arange(-3.,1.,.1)
    rotation_degrees = [ 0, 45, 90, 135, 180, 225, 270, 315 ]

    # Load the data. X is a list of numpy arrays
    # which are the images.
    non_lens_filenames = glob.glob(args['non_lens_glob'])
    lens_filenames = glob.glob(args['lens_glob'])
    filenames = non_lens_filenames + lens_filenames
    X = image_processing.load_images(filenames)
    y = [0] * len(non_lens_filenames) + [1] * len(lens_filenames)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

    X_train, y_train = image_processing.rotate_images( rotation_degrees, X_train, y_train )

    print "len(X_train) =", len(X_train)
    print "len(y_train) =", len(y_train)
    print "len(X_test) =", len(X_test)
    print "len(y_test) =", len(y_test)
    print

    pool = Pool(processes=4)              # start 4 worker processes

    start_time = time.time()
    print 'C_vals: ', C_vals
    print pool.map(score_for_params, C_vals)
    time_taken = time.time() - start_time
    print "Time for 4 worker processes:", time_taken

