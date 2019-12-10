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
style.use('ggplot')

sys.path.append('../utilities/')
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default = "../CSVData/04.13.2019 vs UVA 2.csv")
    parser.add_argument("--output", type =str, default = './')
    opt = parser.parse_args()
    print(opt)
    
    dat = pd.read_csv(opt.csv)
    testDat = subset(dat, ['Pitch Location X', 'Pitch Location Y', 'Strike Type', 'Ball Type', 'Umpire'])
    testDat['Result'] = np.nan
    
    for idx, ball in enumerate(testDat['Ball Type']):
        if(type(ball) == str):
            testDat['Result'][idx] = ball
        else:
            testDat['Result'][idx] = testDat['Strike Type'][idx]
    
    strikes = testDat.loc[testDat['Result'] == 'Take']
    balls = testDat.loc[testDat['Result'] == 'Ball']

    strikes = subset(strikes, ['Pitch Location X', 'Pitch Location Y'])

    balls = subset(balls, ['Pitch Location X', 'Pitch Location Y'])

    dataDict = {-1:np.array(strikes),
           1:np.array(balls)}
    
    frame = testDat.loc[testDat['Result'].isin(['Take', 'Ball'])]
    frame['Result'] = frame['Result'].replace('Take', 1.0)
    frame['Result'] = frame['Result'].replace('Ball', -1.0)
    vec = np.array(subset(frame, ['Pitch Location X', 'Pitch Location Y']))
    label = frame['Result']
    
    