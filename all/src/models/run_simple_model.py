"""Simple logistic regression of a small number of variables.

Reads data from text files.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.cross_validation import train_test_split
from StrongCNN.IO.config_parser import parse_configfile
#from sklearn.model_selection import train_test_split
from _tools import dump_model, load_model, generate_X_scores

def read_data_files(data_files):
    dfs = [pd.read_csv(filename, sep=' ') for filename in data_files]
    for df in dfs:
        assert (df['label'] == dfs[0]['label']).all()

    for idx, df in enumerate(dfs):
        for k in df.keys():
            if k != 'IDs':
                df[k + '_' + str(idx)] = df[k]
                del df[k]

    return reduce(lambda df1, df2 : pd.merge(df1, df2, on='IDs'), dfs)

def get_data_component_scores(data_files):
    data = read_data_files(data_files)
    xcols = [k for k in data.keys() if k.startswith('score')]
    return data[xcols].as_matrix(), data['label_0'].as_matrix(), data['IDs'].as_matrix()

def get_data_avg_scores(data_files):
    data = read_data_files(data_files)
    xcols = [k for k in data.keys() if k.startswith('avgscores')]
    return data[xcols].as_matrix(), data['label_0'].as_matrix(), data['IDs'].as_matrix()

def pickle_model(pklfile, X, y, log_reg_C=10000.) :
    model = LogisticRegression(C=log_reg_C)
    model.fit(X, y)

    dump_model(pklfile, model)

def model_super_scores(pklfile, X, y, IDs, print_auc=True) :
    trained_model = load_model(pklfile)
    IDs, superscores, y = generate_X_scores( model, X, y, IDs )
    if print_auc : 
        from sklearn.metrics import roc_auc_score
        print "AUC from superscores: ",roc_auc_score( y, superscores )
    return IDs, superscores, y

def logistic_train_test_split(X, y, log_reg_C=10000.):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3)
    
    model = LogisticRegression(C=log_reg_C)
    model.fit(X_train, y_train)

    positive_idx = np.where(model.classes_ == 1.)[0][0]

    print "Accuracy on training set:", model.score(X_train, y_train)
    print "AUC on training set:",
    print roc_auc_score(y_train, model.predict_log_proba(X_train)[:, positive_idx])
    print "Confusion matrix on training set:"
    print confusion_matrix(y_train, model.predict(X_train))
    print
    print "Accuracy on test set:", model.score(X_test, y_test)
    print "AUC on test set:",
    print roc_auc_score(y_test, model.predict_log_proba(X_test)[:, positive_idx])
    print "Confusion matrix on test set:"
    print confusion_matrix(y_test, model.predict(X_test))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--cfgdir', 
                        required=True)
    parser.add_argument('data_files', nargs='+')
    parser.add_argument('-C', '--logistic_regression_C', type=float,
                        required=False, default=10000.)
    parser.add_argument('-p', '--pickle_model', 
                        required=False )
    parser.add_argument('-t', '--test_model', 
                        required=False )

    args = vars(parser.parse_args())


    if args['pickle_model'] is not None :
        X, y, _ = get_data_component_scores(args['data_files'])
        print 'Pickling simple model for 16 components'
        print
        pickle_model(cfgdir+'/model_16_component.pkl', X, y, log_reg_C=10000.)
        
        print
        
        X, y, _ = get_data_avg_scores(args['data_files'])
        print 'Pickling simple model for average Scores'
        print
        pickle_model(cfgdir+'/model_avgscore.pkl', X, y, log_reg_C=10000.)


    if args['test_model'] is not None :
        X, y, IDs = get_data_component_scores(args['data_files'])
        print 'Scoring simple 16 component model in '+cfgdir
        print 'on '+args['data_files']
        np.savetxt( cfgdir+'/IDs_scores_16_component.txt',
                    np.asarray(model_super_scores(cfgdir+'/model_16_component.pkl', X, y, IDs)).transpose(),
                    fmt='%s %s %s', header='ID score label')


        X, y, IDs = get_data_avg_scores(args['data_files'])
        print 'Scoring simple average score model in '+cfgdir
        print 'on '+args['data_files']
        np.savetxt( cfgdir+'/IDs_scores_avg.txt',
                    np.asarray(model_super_scores(cfgdir+'/model_avgscore.pkl', X, y, IDs)).transpose(),
                    fmt='%s %s %s', header='ID score label')

        


        
