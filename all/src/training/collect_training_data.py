'''Collect the images as numpy arrays, booleans if each contains an arc or not, and kwargs'''

import IO.read_fits as rf

class CollectTrainingData :
    '''Basic object to collect the training data'''
    def get_test_images(self,datadir) :
        '''Input names of datafiles, e.g. *.fits'''

        from glob import glob
        try :
            print "Collecting the training data... "
            self.fitsobjs = [ rf.get_fits_obj(f) for f in glob(datadir) ]
        except IOError :
            print "Check fits format of these files: ",glob(datadir)
        

    def get_test_classifications(self,classification_list) :
        '''Input list of booleans''' 
        assert( len(self.fitsobj) == len(classification_list) )
        print "Collected classifications for training... "
        return self.classifications = classification_list

    def augment_data(self) :
        '''Will want to figure out how to augment data set (rotations,
        translations, magnifications, etc.) without going overboard in
        memory'''
        print "Augmenting the data for training... "
        pass
