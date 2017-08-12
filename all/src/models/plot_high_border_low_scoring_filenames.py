import StrongCNN.utils.read_model_out as rmo
import StrongCNN.plot_tools.tile_fits as tile_fits

import sys

modeldir = sys.argv[1]

image_types = [ 'true_positives',
                'true_negatives',
                'false_positives',
                'false_negatives',
                'borderline_positives',
                'borderline_negatives',
                ]

for itype in image_types :
    idata = rmo.read_tpr_filenames(modeldir+'/'+itype+'.txt')
    tile_fits.plot_set(idata['fits_file'], modeldir+'/'+itype+'.pdf')


    
    
