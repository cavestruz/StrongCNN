import numpy as np
import sys

def build_parameter_grid( param_grid ) :
    import ast
    return [ {k: ast.literal_eval(v) for k,v in param_grid.iteritems()} ]

def grid_search( pipeline, param_grid, X_train, y_train, X_test, y_test ) :
    '''
    |
    |   Execute a grid search over the parameter grid for a 
    |   given pipeline, and print results to stdout
    |
    '''
    from sklearn.grid_search import GridSearchCV
    from StrongCNN.utils.model_info import print_model_scores
    import time

    gs = GridSearchCV(pipeline, param_grid, n_jobs = 1)

    # Train the model on the training set
    print "Running grid search..."
    start_time = time.time()
    gs.fit(X_train, y_train)
    time_taken = time.time() - start_time
    print "Finished grid search. Took", time_taken, "seconds"
    print
    print "Best score:", gs.best_score_
    print "Best parameters set:",
    best_parameters = gs.best_estimator_.get_params()
    param_names = reduce(lambda x, y : x | y,
                         (p_grid.viewkeys()
                          for p_grid in param_grid))
    for param_name in sorted(param_names):
        print param_name, ":", best_parameters[param_name]
    print

    print "Scores for each parameter combination:"

    #  Splits into a subcycle of something like groups of 3 subsets to
    #  test out the parameters in the grid and select a final best
    #  mean.
    #  http://scikit-learn.org/stable/modules/cross_validation.html
    for grid_score in gs.grid_scores_:
        print grid_score
    print

    print_model_scores(gs, X_train, y_train, X_test, y_test)

