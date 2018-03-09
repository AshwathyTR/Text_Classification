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
import xgboost as xgb
from sklearn.cross_validation import KFold

class Mods:
    
    f = Framework()
    f_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
    }

    # Extra Trees Parameters
    et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
    }

    # AdaBoost parameters
    ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
    }

    # Gradient Boosting parameters
    gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
    }

    # Support Vector Classifier parameters 
    svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
       }
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
    
    def Stack_Method(self):
        train_frame,test_frame= ms.train_test_split(self.f.data,test_size = 0.2, shuffle=True)
        comment_class='toxic'
        ntrain = train_frame.shape[0]
        ntest = test_frame.shape[0]
        kf = KFold(ntrain, n_folds= self.NFOLDS, random_state=self.SEED)
        
        rf = SklearnHelper(clf=RandomForestClassifier, seed=self.SEED, params=self.rf_params)
        et = SklearnHelper(clf=ExtraTreesClassifier, seed=self.SEED, params=self.et_params)
        ada = SklearnHelper(clf=AdaBoostClassifier, seed=self.SEED, params=self.ada_params)
        gb = SklearnHelper(clf=GradientBoostingClassifier, seed=self.SEED, params=self.gb_params)
        svc = SklearnHelper(clf=SVC, seed=self.SEED, params=self.svc_params)
        
        #models=[]
        #for comment_class in tqdm(self.f.classes):
         #   print("training model for %s class"%(self.f.classes))
         # Create our OOF train and test predictions. These base results will be used as new features
        batch = self.f.generate_minibatch(train_frame,15000,0.5,comment_class)
        dataset = self.f.generate_dataset(batch,self.f.data)
        x_chunk=dataset['features']
        y_chunk = dataset[comment_class]
         
        et_oof_train, et_oof_test = self.get_oof(et, x_chunk, y_chunk, test_frame,ntrain,ntest,kf) # Extra Trees
        rf_oof_train, rf_oof_test = self.get_oof(rf,x_chunk, y_chunk, test_frame,ntrain,ntest,kf) # Random Forest
        ada_oof_train, ada_oof_test = self.get_oof(ada, x_chunk, y_chunk, test_frame,ntrain,ntest,kf) # AdaBoost 
        gb_oof_train, gb_oof_test = self.get_oof(gb,x_chunk, y_chunk, test_frame,ntrain,ntest,kf) # Gradient Boost
        svc_oof_train, svc_oof_test = self.get_oof(svc,x_chunk, y_chunk, test_frame,ntrain,ntest,kf) # Support Vector Classifier
        
        #splice answers together to use for second lvl training
        x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
        x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

        print("Training is complete")    
         
        
        gbm = xgb.XGBClassifier(
        #learning_rate = 0.02,
        n_estimators= 2000,
        max_depth= 4,
        min_child_weight= 2,
        #gamma=1,
        gamma=0.9,                        
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread= -1,
        scale_pos_weight=1).fit(x_train, y_chunk)
        predictions = gbm.predict(test_frame)
        submit = pd.DataFrame({ 'comment_text': test_frame['comment_frame'],
                            comment_class: predictions })
        submit.to_csv("Stack_Results.csv", index=False)
        self.f.plot_bias(gbm, test_frame,comment_class)
        
        
        
             #models.append(model)
        #self.f.plot_bias(model, test_frame,comment_class)
     
    # Some useful parameters which will come in handy later on
    SEED = 0 # for reproducibility
    NFOLDS = 5 # set folds for out-of-fold prediction
    
        
    def get_oof(self,clf, x_train, y_train, x_test,ntrain,ntest,kf):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((self.NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]
            
            clf.train(x_tr, y_tr)
            
            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)
            
            oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)    
    

 
m = Mods()
#m.minibatch_LR()
m.Stack_Method()