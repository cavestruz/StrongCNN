import astropy.io.fits as pyfits
import pylab as pl
from scipy.misc import imsave
import sys, glob

def get_fits_obj(fitsfile) :
    '''Returns contents of fitsfile in numpy array format'''
    return pyfits.getdata(fitsfile)

def visualize_fits(fitsfile, figname, colorbar = False) :
    '''Saves fitsfile contents as an image, <figname>.png'''
    pl.figure()
    pl.contour(get_fits_obj(fitsfile))
    if colorbar:
        pl.colorbar()
    pl.savefig(figname+'.png')

def fits_to_png(fitsfile, png_out):
    img_arr = pyfits.getdata(fitsfile)
    imsave(png_out, img_arr)

if __name__ == "__main__" : 
    for fits in glob.glob(sys.argv[1]) :
        pngname = fits[:-4]
        fits_to_png(fits, pngname+'png')
