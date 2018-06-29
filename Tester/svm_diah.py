# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 10:32:38 2018

@author: mr1n17
"""

#from sklearn.linear_model import SGDClassifier
#from main import Framework
from tqdm import tqdm
tqdm.monitor_interval = 0
#import sklearn.model_selection as ms
#import numpy as np
from sklearn.svm import SVC
import pandas as pd
#import LR_utilities as util
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from main import Framework, Test_Suite
from preprocessor import PreProcessor
import sklearn.model_selection as ms
#from pprint import pprint

f = Framework()
t = Test_Suite()
preprocessor = PreProcessor()

### Get Cleaning Dataset level 5

clean_data = preprocessor.clean_all(f.data, 5)
print('done1')
train_frame,test_frame= ms.train_test_split(clean_data,test_size = 0.2, shuffle=True)
print('done2')

test_frame = f.generate_dataset(test_frame, test_frame)
print('done4')

'''we need this for the submission file'''
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [2e-5, 2e5, 2e15], 'class_weight' : ['balanced']},
                   {'kernel': ['linear'], 'C': [2e-5, 2e5, 2e15], 'class_weight' : ['balanced']}]
    

results = []
best_param = {}

for classname in tqdm(class_names):
    batch = f.generate_minibatch(train_frame,250,0.5,classname)

    dataset = f.generate_dataset(batch, train_frame)
    x_chunk=dataset['features']
    y_chunk = dataset[classname]
    y_test = test_frame[classname]
    x_test = test_frame['features']
   
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,  scoring='roc_auc')
    
    clf.fit(x_chunk, y_chunk)

    y_true, y_pred = y_test, clf.predict(x_test)
    print(classname)
    best_param['classname'] = classname
    best_param['best_param'] = clf.best_params_
    best_param['means'] = clf.cv_results_['mean_test_score']
    best_param['stds'] = clf.cv_results_['std_test_score']
    best_param['test_acc'] = roc_auc_score(y_true, y_pred)
    clv2 = SVC()
    clv2.fit(x_chunk, y_chunk)
    y_true, y_pred = y_test, clv2.predict(x_test)
    best_param['default_test_acc'] = roc_auc_score(y_true, y_pred)
    
    print('accuracy for test by using default parameter : ', roc_auc_score(y_true, y_pred))
    results.append(best_param)
    best_param = {}
  

output = pd.DataFrame(results) 
output.to_csv('svm_better_parameter.csv', index=False)

#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#    print()

#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = y_test, clf.predict(x_test)
#    print(roc_auc_score(y_true, y_pred))
#    print()
##clf.fit(x_chunk, y_chunk)
#y_true, y_pred = y_test, clf.predict(x_test)
#print(roc_auc_score(y_true, y_pred))





