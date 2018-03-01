# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:33:45 2018

@author: Ashwathy 
"""
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

class Extractor:
    
    path = r"..\corpora\bad-words.txt"
    
    def get_word_histogram(self, data, vocab_data):
        ''' @params - data :list of comments from which to extract features
            @params - vocab_data: list of comments from which vocabulary should be built
            @output - word vector in sparse matrix form
        '''
        word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                ngram_range=(1, 1),
                max_features=15000)
        word_vectorizer.fit(vocab_data) ##ideally use all_data (where all_data array has to be all_data = pd.concat([train_text, test_text]))
        histogram = word_vectorizer.transform(data)
        return histogram
    
    def num_bad_words(self, data):
        ''' @params - data :list of comments from which to extract features
            @output - list of number of bad words per sentence
        '''
        bad_text=pd.read_csv(self.path,sep="\n", header=None)
        bad_text.columns=['bad']
        bad_words=[]
        for bw in bad_text['bad']:
            bad_words.append(bw)
        feature_val=[]
        for sentence in data:
            count=0
            for bw in bad_words:
                #using regular expressions, each word is seperated by blank or special chars
                if re.search('(\s|^)'+bw+'(\s|$|\.)',sentence):
                    print(bw)
                    count+=1
            feature_val.append(count)
        return feature_val
    
    def num_Upper_Case(self,data):
        ''' @params - data :list of comments from which to extract features
            @output - list of number of upper case characters per sentence
        '''
        feature_val=[]
        for sentence in data:
            count=0
            for word in sentence:
                if word.isupper():
                    count+=1
            feature_val.append(count)
        return feature_val

#  debug testing below

'''

t=Extractor()
ls="I will kill. you","fuck you friend, you are an idiot","HELLO world"
print(t.num_bad_words(ls))
print(t.num_Upper_Case(ls))

'''