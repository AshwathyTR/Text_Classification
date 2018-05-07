# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 17:17:33 2018

@author: Ashwathy
"""
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from main import Framework
from tqdm import tqdm
import sklearn.model_selection as ms
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
import pandas as pd
from sklearn.svm import SVC

#import sys
#sys.path.append('C:\\Users\\Croft\\xgboost\\python-package')

from sklearn.cross_validation import KFold
import xgboost as xgb
from features import Extractor
import matplotlib.pyplot as plt



class Modified_Classifiers:
    
    f = Framework()
    features = Extractor()
    def __init__(self):
        pass
        
    def minibatch_LR(self):
        
        train_frame,test_frame= ms.train_test_split(self.f.data,test_size = 0.35, shuffle=True)
        accuracies={}
        for comment_class in self.f.classes:
            model = SGDClassifier(loss='hinge')
            for i in tqdm(range(0,50,1)):
                batch = self.f.generate_minibatch(train_frame,250,0.5,comment_class)
                dataset = self.f.generate_dataset(batch,self.f.data)
                x_chunk=dataset['features']
                y_chunk = dataset[comment_class]
                model.partial_fit(x_chunk, y_chunk,classes=np.unique(y_chunk))
               #for comment_class in self.f.classes:
            accuracies[comment_class] = self.f.plot_bias(model, test_frame,comment_class)
        return accuracies
    
    def w2v_predict(self,pos,neg, data):
        predictions=[]
        for entry in data:
            if pos.score(entry)>neg.score(entry):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
                
     
    def plot_w2v_bias(self, pos_model, neg_model, test_data, comment_class):
       
        accuracies={}
        test_props = np.arange(0.0, 1.0, 0.1)
        for test_prop in test_props:
                data = self.f.generate_minibatch(test_data,100,test_prop,comment_class)
                predicted = self.w2v_predict(pos_model,neg_model,data['comment_text'])
                accuracy = self.get_accuracy(predicted, test_data[comment_class])
                accuracies[test_prop] = accuracy
        plt.figure(1)
        plt.xlabel('Proportion of clean samples')
        plt.ylabel('accuracy')
        plt.plot(accuracies.keys(),accuracies.values(), 'r^')
        return accuracies
    
    def w2v_classifier(self):
        for class_name in ['toxic']:
             train_frame,test_frame= ms.train_test_split(self.f.data,test_size = 0.2, shuffle=True)
             pos = self.f.get_class_data(train_frame,'toxic',1)
             neg = self.f.get_class_data(train_frame,'toxic',0)
            
            
            
             pos_filename = "..\\corpora\\"+class_name+"_pos.model"
             neg_filename = "..\\corpora\\"+class_name+"_neg.model"
             pos_wv = self.features.get_word_vector_model(pos,pos,pos_filename)
             neg_wv = self.features.get_word_vector_model(neg,neg,neg_filename)
             
             self.plot_w2v_bias(pos_wv,neg_wv, test_frame, 'toxic')
             
             
             
             
c
   
            
            
            
            
            
        
    
    