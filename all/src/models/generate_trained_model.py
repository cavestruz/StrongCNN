import sys, ast
from StrongCNN.IO.config_parser import parse_configfile
from StrongCNN.IO.load_images import load_data
from StrongCNN.IO.augment_data import augment_methods, augment_data
from StrongCNN.pipeline.build_pipeline import build_pipeline
from _tools import train_model, dump_model
import time

cfgdir = sys.argv[1]
cfg = parse_configfile(cfgdir)

start_time = time.time()

# Collect training data
X_train, y_train, _ = load_data(cfg['train_filenames']['non_lens_glob'], 
                                cfg['train_filenames']['lens_glob'])


if 'augment_train_data' in cfg.keys() :
    X_train, y_train = augment_data( X_train, y_train, 
                                     cfg['augment_train_data']['method_label'],
                                     **ast.literal_eval(cfg['augment_train_data']['method_kwargs']))  
    
print "len(X_train) =", len(X_train)
print "len(y_train) =", len(y_train)

print X_train[0].shape


print "train glob ", cfg['train_filenames']['non_lens_glob'], cfg['train_filenames']['lens_glob']

# Build the pipeline
pipeline = build_pipeline(cfg['image_processing'].values(), 
                          cfg['classifier']['label'])

params = {k: ast.literal_eval(v) for k, v in cfg['param_grid'].iteritems()}

train_model(pipeline, X_train, y_train, **params)


dump_model(cfgdir+'/'+cfg['model']['pklfile'], pipeline)

print 'Time taken:', time.time() - start_time

