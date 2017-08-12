''' 
Read in a list of fits files, and tile them
'''
from StrongCNN.IO.load_images import load_images
import matplotlib.pyplot as plt

def plot_set( fits_list, name ) :
    images = load_images( fits_list ) 

    fig, axes = plt.subplots(nrows=len(images)/2, ncols=2)

    for ax, data in zip(axes.flat,images) :
        im = ax.imshow( data )
    fig.subplots_adjust(right=0.9)

    plt.savefig(name)

