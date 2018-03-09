# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 17:17:33 2018

@author: Ashwathy
"""
from sklearn.linear_model import SGDClassifier
from main import Framework
from tqdm import tqdm
import sklearn.model_selection as ms
import numpy as np

class Mods:
    
    f = Framework()
    
    def minibatch_LR(self):
        
        train_frame,test_frame= ms.train_test_split(self.f.data,test_size = 0.2, shuffle=True)
        comment_class='toxic'
        model = SGDClassifier(loss='log')
        for i in tqdm(range(0,50,1)):
            batch = self.f.generate_minibatch(train_frame,100,0.5,comment_class)
            dataset = self.f.generate_dataset(batch,self.f.data)
            x_chunk=dataset['features']
            y_chunk = dataset[comment_class]
            model.partial_fit(x_chunk, y_chunk,classes=np.unique(y_chunk))
            #for comment_class in self.f.classes:
        self.f.plot_bias(model, test_frame,comment_class)
        return model
            
    
 
m = Mods()
m.minibatch_LR()