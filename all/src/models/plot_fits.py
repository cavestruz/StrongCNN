import sys, os, ast
import matplotlib.pyplot as plt
from StrongCNN.IO.load_images import load_images
from StrongCNN.IO.config_parser import parse_configfile
from matplotlib.colors import LogNorm
from StrongCNN.utils.read_model_out import read_failed_ids
from skimage.feature import hog

def read_hog_kwargs(modeldir) :
    cfg = parse_configfile(modeldir) 

    hog_params = {k.split('hog__')[1]: ast.literal_eval(v) for \
                  k, v in cfg['param_grid'].iteritems() \
                  if k.startswith('hog') }    
    return hog_params

def data2plot( fitsfiles, name ) :
    fitsdata = load_images(fitsfiles)
    nrows = len(fitsfiles)/2    
    return name, nrows, fitsdata, [hog(fd, visualise=True, **read_hog_kwargs(modeldir)) for fd in fitsdata]

def multiplot( nrows ) :
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows*6),
                             subplot_kw={'xticks':[],'yticks':[]})
    fig.subplots_adjust(hspace=0.1,wspace=0.05)
    return fig, axes

def multi_imshow( name, nrows, fitsdata, hogdata ) :
    
    fig, axes = multiplot(nrows)

    for ax, d in zip(axes.flat,fitsdata) :
        im = ax.imshow(d, cmap='gray', norm=LogNorm())
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    plt.savefig(imagedir+name+'.pdf')

def multi_hogvisualization( name, nrows, fitsdata, hogdata ) :
    
    fig, axes = multiplot(nrows)

    for ax, d in zip(axes.flat, hogdata) :
        im = ax.imshow(d[1], cmap='gray', norm=LogNorm())
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    plt.savefig(imagedir+name+'_hogvisualization.pdf')
    
def multi_hoghistogram( name, nrows, fitsdata, hogdata ) :
    
    fig, axes = multiplot(nrows)

    for ax, d in zip(axes.flat, hogdata) :
        im = ax.plot(d[0])
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    plt.savefig(imagedir+name+'_hoghistogram.pdf')
    

modeldir = sys.argv[1]
imagedir = modeldir+'/images/'

if not os.path.exists(imagedir) : os.mkdir(imagedir)

# Should return a list of all failed ids
failed_ids = read_failed_ids(modeldir)

lensed_failed_ids = [ fid for fid in failed_ids if 'unlensed' not in fid ]
unlensed_failed_ids = [ fid for fid in failed_ids if 'unlensed' in fid ]

d2p = data2plot(lensed_failed_ids, 'lensed_failed')
multi_imshow(*d2p) 
multi_hogvisualization(*d2p) 
multi_hoghistogram(*d2p)
d2p = data2plot(unlensed_failed_ids, 'unlensed_failed')
multi_imshow(*d2p) 
multi_hogvisualization(*d2p) 
multi_hoghistogram(*d2p)



