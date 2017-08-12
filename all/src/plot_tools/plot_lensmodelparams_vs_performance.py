import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import sys

testdir = sys.argv[1]
paramfile = testdir+'/control_test_params.txt'
control_test = pd.read_csv( paramfile, delim_whitespace=True )

lens_model_params = ['velocity_dispersion', 'ellipticity', 'orientation_angle', 
                     'z', 'magnitude']

def myLogFormat(y,pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)


def plot_ROC() : 
    plt.plot(control_test['fpr'], control_test['tpr'],'b')
    plt.plot(np.arange(0,1.1,.1),np.arange(0,1.1,.1), 'g:')
    plt.xlabel('False Positive Rate',fontsize='xx-large')
    plt.ylabel('True Positive Rate',fontsize='xx-large')
    plt.savefig(testdir+'/ROC_curve.pdf')
    plt.close()

def create_colorbar(cmap, minimum, maximum) :
    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0,0],[0,0]]
    step=(maximum-minimum)/1000.
    levels = np.arange(minimum, maximum,step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)
    plt.clf()
    return CS3

def plot_score_dependence() :
    lensed = control_test[control_test['label'] == 1]
    unlensed = control_test[control_test['label'] == 0]
    for key in control_test.keys() :
        if key not in lens_model_params :
            continue
        for label, marker,color in zip([lensed, unlensed],['d','o'],['g','r']) :
            plt.plot( label[key], label['score'], color=color,
                      marker=marker, ls='None')
            plt.xlabel(key.replace('_',' '),fontsize='xx-large')
            plt.ylabel('score',fontsize='xx-large')
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
            plt.ylim(label['score'].min(),1.2)
            plt.savefig(testdir+'/score_vs_'+key+'_'+marker+'.pdf')
            plt.close()
            
def calculate_colors() :
    '''Create new columns of normalized lens model parameters to feed into the cmap'''
    for key in control_test.keys() :
        if key not in lens_model_params :
            continue
        control_test[key+'_color']=(control_test[key]-control_test[key].min())/(control_test[key].max()-control_test[key].min())

def plot_colorcoded_score_dependence(cmap_kw='hot') :
    calculate_colors()
    lensed = control_test[control_test['label'] == 1]
    unlensed = control_test[control_test['label'] == 0]
    cmap=plt.get_cmap(cmap_kw)

    for key in control_test.keys() :
        if key not in lens_model_params :
            continue
        for color_key in lens_model_params :
            if color_key == key : continue
            for label,marker in zip([lensed, unlensed],['d','o']) :
                plt.scatter( label[key], label['score'], 
                             color=cmap(label[color_key+'_color']),
                             vmin=label[color_key].min(),vmax=label[color_key].max(),
                             marker=marker,cmap=cmap)
                plt.xlabel(key.replace('_',' '),fontsize='xx-large')
                plt.ylabel('Score',fontsize='xx-large')
                plt.yscale('log')
                plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
                plt.ylim(label['score'].min(),1.2)
                CS3 = create_colorbar(cmap, label[color_key].min(), label[color_key].max())
                plt.colorbar(CS3)
                plt.savefig(testdir+'/score_vs_'+key+'_'+marker+'_colorcodedby_'+color_key+'.pdf')
                plt.close()
#plot_ROC()
#plot_score_dependence()
plot_colorcoded_score_dependence()
