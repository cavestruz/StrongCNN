import numpy as np
from collections import Counter
from model_info import get_scores, confusion_matrix

def get_false_predictions_list( trained_model, X, y, filenames ) :
    '''
    |    Trained model must have predict and score methods.
    |    X, y and filenames must have same length
    '''
    assert( len(X) == len(y) ) 
    assert( len(y) == len(filenames) )

    # Score the test set                                                                
    print "Confusion matrix on test set"
    print confusion_matrix(trained_model.predict(X), y)
    print
    print "Score on test set =", trained_model.score(X, y)


    successful_predictions = map( lambda x: x[0] == x[1], 
                                  zip( trained_model.predict( X ), y) )
    
    return [ sf[1] for sf in zip( successful_predictions, filenames ) if sf[0] == 0 ]

def get_ranked_predictions( trained_model, X, y, filenames ):
    '''
    | Returns a list of tuples giving the scores for each
    | filename that is sorted descending by score.
    | trained_model must have predict_proba or
    | decision_function method.
    |
    | Output format:
    |
    | [(filename1, score1, label1),
    |  (filename2, score2, label2),
    |  ...]
    |
    | where filename? is the filename corresponding to
    | a given row of X from the filenames list, score?
    | is the score for the given row of X from
    | trained_model, and label? is the 1/0 label for
    | the given row of X taken from the array y. The
    | output will be sorted such that
    | score1 >= score2 >= score3 >= ...
    '''
    assert( len(X) == len(y) )
    assert( len(y) == len(filenames) )
    
    scores = get_scores(trained_model, X)
    return sorted(zip(filenames, scores, y),
                  key = lambda (filename, score, label) : score,
                  reverse = True)

def _calc_tpr_fpr( scores, labels ) :
    '''
    |    Runs down the ordered scores and returns tpr, fpr
    |    corresponding to varied thresholds
    |
    |    Note: the built in roc_curve from sklearn.metrics only
    |    returns the necessary tpr, fpr values to make the plot, and
    |    not the intermediate points for constant tpr or fpr.
    '''
    assert( len(scores) == len(labels) )
    #assert( all( scores[i] >= scores[i+1] for i in xrange(len(scores)-1) )  )

    tpr, fpr = [], []
    num_label1 = float(Counter(labels)[1])
    num_label0 = float(Counter(labels)[0])

    for i in range(len(labels)) :
        tpr.append(Counter(labels[:i])[1] / num_label1)
        fpr.append(Counter(labels[:i])[0] / num_label0)

    return np.array(tpr), np.array(fpr)

def _get_tpr_fpr_indices( tpr, tpr_min, tpr_max, fpr, fpr_min, fpr_max ) :
    '''
    |    Return the indices corresponding to tpr min and max, fpr_min and fpr_max
    '''

    assert( tpr_min <= tpr[0] and tpr_max >= tpr[-1] ) 
    assert( fpr_min <= fpr[0] and fpr_max >= fpr[-1] ) 
    assert( tpr_min <= tpr_max and fpr_min <= fpr_max ) 

    tpr_indices = np.where( (tpr <= tpr_max) & (tpr >= tpr_min) )
    fpr_indices = np.where( (fpr <= fpr_max) & (fpr >= fpr_min) ) 

    return tpr_indices, fpr_indices

def get_filenames_in_threshold_range( trained_model, X, y, filenames, (tpr_min,tpr_max), (fpr_min, fpr_max) ) :
    '''
    |    Return the filenames and corresponding scores, labels, tpr, and fpr that satisfy
    |    the tpr or fpr range by broadcasting.  
    |
    |    Output format:
    |    [(filename1, score1, label1, tpr1, fpr1), (filename2, score2, label2, tpr2, fpr2), ...]
    |    where the data corresponding to the filenames fell within the tpr_min/max range
    |    [(filename1b, label1b), (filename2b, label2b), ...]
    |    where the data corresponding to the filenames fell within the fpr_min/max range
    '''
    
    ranked_predictions = get_ranked_predictions( trained_model, X, y, filenames )               
    ordered_filenames = np.array([ rp[0] for rp in ranked_predictions ])
    ordered_scores = np.array([ rp[1] for rp in ranked_predictions ])
    ordered_labels = np.array([ rp[2] for rp in ranked_predictions ])
                   
    tpr, fpr = _calc_tpr_fpr( ordered_scores, ordered_labels )
    tpr_indices, fpr_indices = _get_tpr_fpr_indices( tpr, tpr_min, tpr_max, fpr, fpr_min, fpr_max ) 

    return zip( ordered_filenames[tpr_indices], ordered_scores[tpr_indices], ordered_labels[tpr_indices], tpr[tpr_indices], fpr[tpr_indices] ), \
        zip( ordered_filenames[fpr_indices], ordered_scores[fpr_indices], ordered_labels[fpr_indices], tpr[fpr_indices], fpr[fpr_indices] )


