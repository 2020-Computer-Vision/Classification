from sklearn import *
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import preprocessing
from matplotlib import style
import argparse
import matplotlib.patches as mpatches
style.use('ggplot')

sys.path.append('../utilities/')
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default = "../CSVData/04.13.2019vsUVAGM2.csv", help = "file path to CSV from game")
    parser.add_argument("--output", type =str, default = '../output/default.pdf', help = "file path to where we will output")
    parser.add_argument("--name", type=str, default = "Game Chart", help = "What the User wishes to call the plot outputted.")
    opt = parser.parse_args()
    print(opt)
    
    # read in our csv as dat
    dat = pd.read_csv(opt.csv)
    
    # reduce our dataframe to what we care about for SVD
    dat = subset(dat, ['Pitch Location X', 'Pitch Location Y', 'Strike Type', 'Ball Type', 'Umpire'])
    
    # Create a result vector containing the balls and strikes label.
    dat['Result'] = dat['Strike Type']
    dat['Result'].fillna("Ball", inplace = True)
    
    # Reduce our data to only the pitches the umpire calls
    frame = dat[dat.Result.isin(['Take', 'Ball'])].copy()
    frame['Result'].replace('Take', 1.0, inplace = True)
    frame['Result'].replace('Ball', -1.0, inplace = True)
    vec = np.array(subset(frame, ['Pitch Location X', 'Pitch Location Y']))
    
    # Run our model
    clf = svm.SVC(gamma='scale', cache_size=7000, kernel='rbf')
    clf.fit(vec, frame.Result)

    

'''
Helper Function to plot our SVD

@param model is the model we are plotting
@param ax is the figure that we are plotting
'''
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = [-10, 240]
    ylim = [250, -10]
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='purple',
               levels=[0], alpha=0.6,
               linestyles=['-'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.hlines(y=50, xmin=60, xmax=185, linewidth=2, color='g', alpha = 0.3)
    ax.hlines(y=190, xmin=60, xmax=185, linewidth=2, color='g', alpha = 0.3)
    ax.vlines(x=60, ymin=50, ymax=190, linewidth=2, color='g', alpha = 0.3)
    ax.vlines(x=185, ymin=50, ymax=190, linewidth=2, color='g', alpha = 0.3)

# Save our Figure
plt.scatter(vec[:, 0], vec[:, 1], c=frame.Result, s=50, cmap='bwr')
plt.xlabel('XOS X Value')
plt.ylabel('XOS Y Value')
plt.title(opt.name)
red_patch = mpatches.Patch(color='red', label='Strikes')
blue_patch = mpatches.Patch(color='blue', label='Balls')
green_patch = mpatches.Patch(color='green', label='Strike Zone')
purple_patch = mpatches.Patch(color='purple', label='Predicted Strike Zone')
lgd = plt.legend(handles=[red_patch, blue_patch, green_patch, purple_patch], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plot_svc_decision_function(clf)
plt.savefig(opt.output, bbox_extra_artists=(lgd,), bbox_inches='tight')
    