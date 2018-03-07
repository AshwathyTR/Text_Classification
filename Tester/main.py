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
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from tqdm import tqdm
from preprocessor import PreProcessor
from scipy.sparse import hstack
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Framework:
    
    data=[]
    classes=[]
    path = r"..\Toxic Comment Data\train.csv"
    data_repo = r"..\Toxic Comment Data"
    feature_extractor = Extractor()
    
    def __init__(self):
        self.data = pd.read_csv(self.path);
        self.classes = self.data.keys()[2:]
        
    
    def generate_dataset(self,data, vocab_data):
        ''' @params - data :dataframe containing list of comments from which to extract features and corresponding classifications
            @params - vocab_data: dataframe containing list of comments from which vocabulary should be built
            @output - dictionary containing features, comment_text and classification targets
        '''
        dataset = {}
        word_vectors = self.feature_extractor.get_word_histogram(data['comment_text'],vocab_data['comment_text'])
        #bad_words_vectors=self.feature_extractor.num_bad_words(data['comment_text'])
        #dataset['features'] =  hstack((word_vectors,bad_words_vectors))
        dataset['features'] = word_vectors
        dataset['comment_text'] = data['comment_text']
        for classname in self.classes:
            dataset[classname] = data[classname]
        return dataset
    
    def generate_train_test(self,data):
        ''' @params - data :dataframe containing list of comments to split into training and test sets
            @output - train and test sets in the form of dictionary containing features, comment_text and classification targets
        '''
        train_frame,test_frame= ms.train_test_split(data,test_size = 0.2, shuffle=True)
        train = self.generate_dataset(train_frame,data)
        test = self.generate_dataset(test_frame,data)
        return train,test
    
    def generate_minibatch(self,data, chunksize, clean_prop, comment_class):
        ''' @params - data :dataframe from which to generate minibatch
            @params - chunksize: size of mini batch
            @params - clean_prop: proportion of clean samples in minibatch
            @params - comment_class: which comment_class to split on
            @output - dataframe containing specific concentration of clean samples
        '''
        positive = data.loc[self.data[comment_class] == 1]
        negative = data.loc[self.data[comment_class] == 0]
        num_neg = int(chunksize*clean_prop)
        num_pos = chunksize - num_neg
        sample_pos = positive.sample(n=num_pos)
        sample_neg = negative.sample(n=num_neg)
        sample = pd.concat([sample_pos,sample_neg])
        sample=shuffle(sample)
        return sample        
     
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
            if((float(pred)>0.5 and int(corr) == 1)or(float(pred)<0.5 and int(corr) == 0)):
                results.append(0)
            else:
                results.append(1)
        return results
    
    def get_accuracy(self,predictions, correct):
        ''' @params - predictions: list of predictions made ny classifier
            @params - correct: correct values from dataset
            @output - accuracy
        '''
        right=0
        wrong=0
        
        for pred, corr in tqdm(zip(predictions,correct)):
            if(pred == corr) :
                right+=1
            else:
                wrong+=1
        accuracy = float(right)/float(right+wrong)
        return accuracy
    
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
    
    def plot_bias(self, model, test_data, comment_class):
        ''' @params - model: trained classifier
            @params - test_data: test data
            @params - comment_class : class on which to test bias
            @output - dict containing accuracies at different concentrations of clean data and plot
        '''
        accuracies={}
        test_props = np.arange(0.0, 1.0, 0.1)
        for test_prop in test_props:
                data = self.generate_minibatch(test_data,500,test_prop,comment_class)
                dataset = self.generate_dataset(data,self.framework.data)
                predicted = model.predict(dataset['features'])
                accuracy = self.get_accuracy(predicted, dataset[comment_class])
                accuracies[test_prop] = accuracy
        plt.figure(1)
        plt.xlabel('Proportion of clean samples')
        plt.ylabel('accuracy')
        plt.plot(accuracies.keys(),accuracies.values(), 'r^')
        return accuracies
        
 

class Test_Suite:
    
    framework = Framework()
    preprocessor=PreProcessor()
    
    def __init__(self):
        self.framework.__init__()
        
    def run(self):
        ''' Main fn: example to show how the framework should be used
        '''
        self.framework.data['comment_text']=self.preprocessor.clean_data(self.framework.data['comment_text'])
        dataset = self.framework.generate_dataset(self.framework.data,self.framework.data)
        classifier = LogisticRegression(solver='sag')
        scores = self.framework.get_scores(classifier, dataset)
        print(scores)

        train,test = self.framework.generate_train_test(self.framework.data)
        output = self.framework.get_output(classifier,train,test)
        output.to_csv('output.csv', index=False)
        
    def clean_compare(self):
        ''' Tries out different levels of cleaning and outputs results
        '''
        scores={}
        for clean_level in tqdm(range(5,-1,-1)):
            clean_data = self.preprocessor.clean_all(self.framework.data, clean_level)
            dataset = self.framework.generate_dataset(clean_data, clean_data)
            classifier = LogisticRegression(solver='sag')
            scores[str(clean_level)] = self.framework.get_scores(classifier, dataset)
        scoresframe = pd.DataFrame.from_dict(scores)
        scoresframe.to_csv('cleaning_comparision.csv', index=False)
        return scoresframe

       
    def classifier_compare(self):
        ''' Tries out different classifiers and outputs results
        '''
        scores={}
        clean_data = self.preprocessor.clean_all(self.framework.data, 5)
        dataset = self.framework.generate_dataset(clean_data,clean_data)
        classifier = LogisticRegression(solver='sag')
        scores["LR"] = self.framework.get_scores(classifier,dataset)
        classifier = SGDClassifier()
        scores["SGD"] = self.framework.get_scores(classifier,dataset)
        classifier = SVC()
        scores["SVC"] = self.framework.get_scores(classifier,dataset)
        scoresframe = pd.DataFrame.from_dict(scores)
        scoresframe.to_csv('classifier_comparision.csv', index=False)
        return scoresframe
    
    
    def bias_check(self):
        ''' Checks how accuracy varies with varying concentration of clean samples
        '''
        classifier = LogisticRegression(solver='sag')
        train_frame,test_frame= ms.train_test_split(self.framework.data,test_size = 0.2, shuffle=True)
        train = self.framework.generate_dataset(train_frame,self.framework.data)
        classifier.fit(train['features'], train['toxic'])
        self.framework.plot_bias(classifier, test_frame, 'toxic')
