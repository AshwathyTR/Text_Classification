"""
Created on Wed Feb 14 20:37:57 2018

@author: Ashwathy T Revi
@description: csv parser functions
"""


import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from features import Extractor
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm




class Tester:
    
    data=[]
    classes=[]
    path = r"..\Toxic Comment Data\train.csv"
    feature_extractor = Extractor()
    
    def __init__(self):
        self.data = pd.read_csv(self.path);
        self.classes = self.data.keys()[2:7]
        
    
    def generate_dataset(self,data, vocab_data):
        ''' @params - data :list of comments from which to extract features
            @params - vocab_data: list of comments from which vocabulary should be built
            @output - dictionary containing features, comment_text and classification targets
        '''
        dataset = {}
        word_vectors = self.feature_extractor.get_word_histogram(data['comment_text'],vocab_data['comment_text'])
        dataset['features'] = word_vectors
        dataset['comment_text'] = data['comment_text']
        for classname in self.classes:
            dataset[classname] = data[classname]
        return dataset
    
    def generate_train_test(self,data):
        ''' @params - data :list of comments to split into training and test sets
            @output - train and test sets in the form of dictionary containing features, comment_text and classification targets
        '''
        train_frame,test_frame= ms.train_test_split(data,test_size = 0.2, shuffle=True)
        train = self.generate_dataset(train_frame,data)
        test = self.generate_dataset(test_frame,data)
        return train,test
     
    def get_scores(self,classifier,dataset):
        ''' @params - classifier :classifier from sklearn
            @output - dict containing score of classifier per class
        '''
        scores = {}
        for class_name in tqdm(self.classes):
               train_target = dataset[class_name]
               train_input = dataset['features']
               cv_loss = np.mean(ms.cross_val_score(classifier, train_input, train_target, cv=3, scoring='roc_auc'))
               scores[class_name] = cv_loss
        return scores
    
    def check_result(self,predictions, correct):
        ''' @params - predictions: list of predictions made ny classifier
            @params - correct: correct values from dataset
            @output - list that says 1 or 0 to indicate if a mistake was made or not
        '''
        results=[]
        for pred, corr in tqdm(zip(predictions,correct)):
            if((int(pred)>0.5 and int(corr) == 1)or(int(pred)<0.5 and int(corr) == 0)):
                results.append(0)
            else:
                results.append(1)
        return results
    
    def get_output(self,classifier,train,test):
        ''' @params - classifier :classifier from sklearn
            @params - train,test : training and test dicts with features, original comment and classification
            @output - dataframe with output from the classifier (also written to csv file)
        '''
        output={}
        output['comment_text']=test['comment_text']
        for class_name in tqdm(self.classes):
            classifier.fit(train['features'], train[class_name])
            output[class_name+'_predictions'] = classifier.predict_proba(test['features'])[:, 1]
            output[class_name+'_real'] = test[class_name]
            output[class_name+'_mistake'] = self.check_result(output[class_name+'_predictions'],output[class_name+'_real'])
        output_frame = pd.DataFrame.from_dict(output)
        return output_frame
    
    def run(self):
        ''' Main fn: runs the test
        '''
        dataset = self.generate_dataset(self.data,self.data)
        classifier = LogisticRegression(solver='sag')
        scores = self.get_scores(classifier, dataset)
        print scores
        
        train,test = self.generate_train_test(self.data)
        output = self.get_output(classifier,train,test)
        output.to_csv('output.csv', index=False)
        
t = Tester()
t.run()