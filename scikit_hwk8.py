#!/usr/bin/env python

def thetimestamp ():

    now = datetime.datetime.now()
    thetimestamp = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
    return thetimestamp


from argparse import ArgumentParser
from sklearn.datasets import load_boston

import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotly.plotly as py
import plotly.tools as tls

#import seaborn as sns
#sns.set(style="darkgrid")


data = load_boston()

x = pd.DataFrame(data.data)
x.columns = data.feature_names
y=pd.DataFrame(data.target)
y.columns=['MEDV']

data = pd.concat([x, y], axis=1)


for c in data.columns:
    ####print(c)           # Returns name
    ####print(data[c])     # Returns data

    for d in data.columns:

        for e in data.columns:

            #for f in data.columns:

                #if (c != d) and (c !=e ) and (c != f) and (d != e) and (d != f) and (e != f):

                if (c != d) and (c !=e ) and  (d != e) :
                    ########## Plotter Data Start

                    plt.figure(figsize=(12,8))
                    plt.title("Boston Housing Data")

                    sorted_dataframe = data.sort_values(c) # where c was the first column we were plotting
                    sorted_dataframe = sorted_dataframe.reset_index(drop=True)

                    #plt.plot( data[c], 'r',label=c + ' : Mn(' + str(np.round(np.mean(data[c]),decimals=2)) + '), Std(' + str(np.round(np.std(data[c]),decimals=2)) + ')')
                    plt.plot( sorted_dataframe[c], 'r',label=c + ' : Mn(' + str(np.round(np.mean(data[c]),decimals=2)) + '), Std(' + str(np.round(np.std(data[c]),decimals=2)) + ')')
                    plt.plot( data[d], 'b',label=d + ' : Mn(' + str(np.round(np.mean(data[d]),decimals=2)) + '), Std(' + str(np.round(np.std(data[d]),decimals=2)) + ')')
                    plt.plot( data[e], 'g',label=e + ' : Mn(' + str(np.round(np.mean(data[e]),decimals=2)) + '), Std(' + str(np.round(np.std(data[e]),decimals=2)) + ')')
                    #plt.plot( data[f], 'c',label=f + ' : Mn(' + str(np.round(np.mean(data[f]),decimals=2)) + '), Std(' + str(np.round(np.std(data[f]),decimals=2)) + ')')

                    plt.grid(True)
                    plt.xlabel('Respondents')
                    plt.ylabel('Units')
                    plt.legend(loc='upper left')

                    plt.savefig('PLT_' + c + '_' + d + '_'+ e + '_' + thetimestamp() + '.png')
                    #plt.savefig('PLT_' + c + '_' + d + '_'+ e + '_' + f + '_' + thetimestamp() + '.png')
                    plt.close()
                    ########## Plotter Data End


###Add 2 more rows to the Dataframe with the Mean and the Standard Deviation
data.loc['MEAN'] = np.round(np.mean(data),decimals=2)
data.loc['STD'] = np.round(np.std(data),decimals=2)
print(data)



print('\n\n\n -> Completed')
