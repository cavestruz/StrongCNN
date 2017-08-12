def read_failed_ids(modeldir) :
    for line in open(modeldir+'/model_test.out') :
        if line.startswith('[') :
            import ast
            return ast.literal_eval(line)


def read_tpr_filenames( filestxt ) :
    import numpy as np 

    return np.loadtxt(filestxt, 
                      dtype={'names':('fits_file','score','label','tpr','fpr'), 
                             'formats': ('|S150',np.float, np.int, np.float, np.float)})
