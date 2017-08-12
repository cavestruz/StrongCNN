import numpy as np
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

def confusion_matrix(predicted, actual):
    '''
    | Outputs (what model thinks it is, what it is)

    '''
    predicted, actual = map(np.array, [predicted, actual])
    return Counter(zip(predicted, actual))

def print_model_scores(model, X_train, y_train, X_test, y_test) :
    ''' 
    | Outputs scores from training and testing sets
    | The input model arg must have predict and score as methods
    '''


    # Show score for training set using best parameters
    # Confusion matrix is 
    print "Confusion matrix on training set"
    print confusion_matrix(model.predict(X_train), y_train)
    print
    print "Score on training set =", model.score(X_train, y_train)
    print

    # Score the test set
    print "Confusion matrix on test set"
    print confusion_matrix(model.predict(X_test), y_test)
    print
    print "Score on test set =", model.score(X_test, y_test)

def roc_curve_data(model, X, y):
    '''
    | Outputs the ROC curve for the given model and data.
    | model must have predict_proba or decision_function
    | method.
    |
    | The output is two arrays:
    | 1. fpr = False positive rate
    | 2. tpr = True positive rate
    |
    | A ROC plot shows fpr on the x axis and tpr on the
    | y axis.  More info:
    | http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    '''
    scores = get_scores(model, X)
    fpr, tpr, _ = roc_curve(y, scores)
    return fpr, tpr

def roc_curve_plot(model, X, y, outfile):
    '''
    | Plots the ROC curve for the given model and data.
    | Uses roc_curve_data to get the points for the plot,
    | and then saves a pdf of the plot to outfile. Also
    | returns the data points in the plot in two arrays:
    | fpr, tpr.
    '''
    fpr, tpr = roc_curve_data(model, X, y)
    plt.clf()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle = '--', color = 'k')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(outfile)
    return fpr, tpr

def roc_auc(model, X, y):
    '''
    | Outputs the area under the ROC curve for the given
    | model and data.  model must have predict_proba or
    | decision_function method.
    | The output is the area under the ROC curve (AUC).
    | More info:
    | http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    '''
    scores = get_scores(model, X)
    return roc_auc_score(y, scores)

def model_coeff_plot(model, outfile):
    '''
    | Plots the coefficents of the model and saves to
    | outfile.  model must have model.coef_ attribute.
    | Also returns the array of coefficients that are
    | plotted.
    '''
    data = np.reshape(model.coef_, (model.coef_.shape[1],))
    plt.clf()
    plt.plot(data)
    plt.savefig(outfile)
    return data

def get_scores(model, X):
    '''
    | Returns the scores for each point. model must
    | have predict_proba or decision_function method.
    '''
    index_of_positive_class = np.where(model.classes_ == 1)[0][0]
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:,index_of_positive_class]
    elif hasattr(model, 'decision_function'):
        return model.decision_function(X)
    else:
        raise Exception("model must have method predict_proba or decision_function")

def generate_AUC_fpr_tpr( y, predictions ) :
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve( y, predictions )
    return metrics.auc(fpr, tpr), (fpr, tpr)
