# -*- coding: utf-8 -*-
"""
Created on Wed May 09 16:37:18 2018

@author: hp
"""

from main import Framework
f = Framework()
f.__init__()
import numpy as np

from results import mini_batch_svm,balanced_svm,clean_compare,baseline_svm,mini_batch_glove,histogram_rbf,glove_rbf

import matplotlib.pyplot as plt
def plot(dicts):
    for classname in f.classes:
        fig = plt.figure()
        for entry in dicts.keys():
            mean_dict = get_mean_dict(dicts[entry])
            lists = sorted(mean_dict[classname].items())
            x, y = zip(*lists) 
            plt.plot(x, y,label=entry)
        fig.suptitle(classname, fontsize=20)
        plt.xlabel('proportion of clean samples', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        fig.legend()
        fig.savefig(classname+'_bias.jpg')
            
        
        
def get_mean_dict(runs):
    mean_dict={}
    biases=runs[0]['toxic'].keys()
    for classname in f.classes:
        mean_dict[classname]={}
    for classname in f.classes:
       for bias in biases:
           mean_dict[classname][bias] = np.mean([float(runs[run][classname][bias]) for run in runs.keys()])
    return mean_dict

def get_run_list(runs):
    mean_dict={}
    biases=runs[0]['toxic'].keys()
    for classname in f.classes:
        mean_dict[classname]={}
    for classname in f.classes:
       for bias in biases:
           mean_dict[classname][bias] = [float(runs[run][classname][bias]) for run in runs.keys()]
    return mean_dict
    


plot({'histogram_rbf':histogram_rbf, 'glove_rbf':glove_rbf, 'baseline_linear_svm':baseline_svm,'linear_glove':mini_batch_glove })
           
