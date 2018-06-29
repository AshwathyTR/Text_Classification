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
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from tqdm import tqdm
from preprocessor import PreProcessor
from scipy.sparse import hstack
from sklearn.utils import shuffle, class_weight
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
#from shogun import *
#from modshogun import StringCharFeatures, RAWBYTE
#from shogun.Kernel import SSKStringKernel
#from libsvm import *
print '0'
class Framework:
    
    data=[]
    classes=[]
    path = r"..\Toxic Comment Data\train.csv"
    data_repo = r"..\Toxic Comment Data"
    feature_extractor = Extractor()
    
    def __init__(self):
        self.data = pd.read_csv(self.path);
        self.classes = self.data.keys()[2:]
        self.preprocessor = PreProcessor()
    
    def generate_dataset(self,data, vocab_data):
        ''' @params - data :dataframe containing list of comments from which to extract features and corresponding classifications
            @params - vocab_data: dataframe containing list of comments from which vocabulary should be built
            @output - dictionary containing features, comment_text and classification targets
        '''
        dataset = {}
        #vocab = self.preprocessor.clean_all(vocab_data,-3)
        #train = self.preprocessor.clean_all(data,3)
        vocab = vocab_data['comment_text']
        train = data['comment_text']
        word_vectors = self.feature_extractor.get_word_vectors(train,vocab)
        #word_vectors = self.feature_extractor.get_word_histogram(train,vocab)
        #bad_words_vectors=self.feature_extractor.num_bad_words(data['comment_text'])
        #dataset['features'] =  hstack((word_vectors,bad_words_vectors))
        dataset['features'] = word_vectors
        
        dataset['comment_text'] = train
        for classname in self.classes:
            dataset[classname] = data[classname]
        return dataset
    
    def generate_train_test(self,data):
        ''' @params - data :dataframe containing list of comments to split into training and test sets
            @output - train and test sets in the form of dictionary containing features, comment_text and classification targets
        '''
        train_frame,test_frame= ms.train_test_split(data,test_size = 0.35, shuffle=True)
        train = self.generate_dataset(train_frame,data)
        test = self.generate_dataset(test_frame,data)
        return train,test
    
    def get_class_data(self,data,class_name,positive):
        df = data[data[class_name] == positive]
        return df
        
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
        '''
            @params - model: trained classifier
            @params - test_data: test dataframe
            @params - comment_class : class on which to test bias
            @output - dict containing accuracies at different concentrations of clean data and plot
        '''
        accuracies={}
        test_props = np.arange(0.0, 1.0, 0.1)
        rbf_feature = RBFSampler()
        for test_prop in test_props:
                data = self.generate_minibatch(test_data,100,test_prop,comment_class)
                dataset = self.generate_dataset(data,self.data)
                x_chunk=dataset['features']
                X_features = rbf_feature.fit_transform(x_chunk)
                predicted = model.predict(X_features)
                accuracy = self.get_accuracy(predicted, dataset[comment_class])
                accuracies[test_prop] = accuracy
        plt.figure(1)
        plt.xlabel('Proportion of clean samples')
        plt.ylabel('accuracy')
        plt.plot(accuracies.keys(),accuracies.values(), 'r^')
        return accuracies
    
    def box_plot(self,model,test_data):
        
        accuracies=[]
        for comment_class in self.classes:
            class_accuracies=[]
            for run in range(1,5,1):
                data = self.generate_minibatch(test_data,100,0.5,comment_class)
                dataset = self.generate_dataset(data,self.data)
                predicted = model.predict(dataset['features'])
                accuracy = self.get_accuracy(predicted, dataset[comment_class])
                class_accuracies.append(accuracy)
            accuracies.append(class_accuracies)
            
        plt.figure(1)
        plt.xlabel('Class')
        plt.ylabel('accuracy')
        plt.boxplot(accuracies)
        plt.xticks ([1,2,3,4,5,6], self.classes)
        return accuracies
        
 

class Test_Suite:
    
    framework = Framework()
    preprocessor=PreProcessor()

    def __init__(self):
        self.framework.__init__()
        
    def run(self):
        ''' Main fn: example to show how the framework should be used
        '''
        #self.framework.data=self.preprocessor.clean_all(self.framework.data,6)
        dataset = self.framework.generate_dataset(self.framework.data,self.framework.data)
        classifier = LogisticRegression(solver='sag')
        scores = self.framework.get_scores(classifier, dataset)
        print(scores)

        #train,test = self.framework.generate_train_test(self.framework.data)
        #output = self.framework.get_output(classifier,train,test)
        #output.to_csv('output.csv', index=False)
        
    def balanced_svm(self,dataset, classname):
        '''
        Run Linear SVM with extra penalty for False negatives

        '''
        classifier = LinearSVC(class_weight='balanced',verbose=1, max_iter=5000)
        classifier.fit(dataset['features'],dataset[classname])
        return classifier
    
    def clean_compare(self,i):
        ''' Tries out different levels of cleaning and outputs results
        '''
        scores={}
        for clean_level in tqdm(range(6,-1,-1)):
            clean_data=self.framework.data
            clean_data['comment_text'] = self.preprocessor.clean_all(self.framework.data, clean_level)
            
            classwise_scores={}
            for classname in self.framework.classes:
                 train_frame,test_frame= ms.train_test_split(clean_data,test_size = 0.35, shuffle=True,stratify=clean_data[classname])
                 train_set = self.framework.generate_dataset(train_frame,clean_data)
       
                 model = self.balanced_svm(train_set,classname)
                 test_sample = self.framework.generate_minibatch(test_frame,100,0.5,classname)
                 test_set = self.framework.generate_dataset(test_sample,clean_data)
                 predicted = model.predict(test_set['features'])
                 accuracy = self.framework.get_accuracy(predicted, test_set[classname])
                 classwise_scores[classname] = accuracy
            scores[str(clean_level)] = classwise_scores
        scoresframe = pd.DataFrame.from_dict(scores)
        scoresframe.to_csv('cleaning_comparision'+str(i)+'.csv')
        return scoresframe

       
    def classifier_compare(self,i):
        ''' Tries out different classifiers and outputs results
        '''
        scores={}
        #clean_data = self.preprocessor.clean_all(self.framework.data, 5)
        dataset = self.framework.generate_dataset(self.framework.data,self.framework.data)
        classifier = LogisticRegression(solver='sag')
        scores["LR"] = self.framework.get_scores(classifier,dataset)
        classifier = LinearSVC(verbose=1)
        scores["SVC"] = self.framework.get_scores(classifier,dataset)
        scoresframe = pd.DataFrame.from_dict(scores)
        scoresframe.to_csv('classifier_comparision'+str(i)+'.csv', index=False)
        return scoresframe
    
    
    
    '''def string_kernel(self):
        clean_data = self.preprocessor.clean_all(self.framework.data, 5)
        train,test = self.framework.generate_train_test(clean_data)
        #  n=1  0.5=lambda Lodhi 2002
        string_kernel = SSKStringKernel(train['features'], train['features'], 1, 0.5)
        target = BinaryLabels(train['toxic'])
        c_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(train),
                                                 train)
        param = svm_parameter(weight = c_weight)
        classifier = LibSVM(1.0, string_kernel, target)
        classifier.train()
        predicted = classifier.apply(test['features']).get_labels()
        print (self.framework.get_accuracy(predicted,test['toxic']))'''
        
        
    def bias_check(self):
        ''' Checks how accuracy varies with varying concentration of clean samples
        '''
        accuracies={}
        clean_data=self.framework.data
        #clean_data['comment_text'] = self.preprocessor.clean_all(self.framework.data, 5)
        for class_name in self.framework.classes:
            classifier = LinearSVC(verbose=1)
            train_frame,test_frame= ms.train_test_split(clean_data,test_size = 0.35, shuffle=True)
            train = self.framework.generate_dataset(train_frame,self.framework.data)
            classifier.fit(train['features'], train[class_name])
            accuracies[class_name] = self.framework.plot_bias(classifier, test_frame, class_name)
        return accuracies
        
    def run_balanced_svm(self):
        accuracies ={}
        for class_name in self.framework.classes:
            train_frame,test_frame= ms.train_test_split(self.framework.data,test_size = 0.35, shuffle=True,stratify=self.framework.data['toxic'])
            train_set = self.framework.generate_dataset(train_frame,self.framework.data)
            model = self.balanced_svm(train_set,class_name)
            accuracies[class_name] = self.framework.plot_bias(model,test_frame,class_name)        
        return accuracies
    
      
    
    

'''mc = Test_Suite()
acc={}
for i in range(1,6,1):
    acc[i] = mc.run_balanced_svm()

print acc'''
