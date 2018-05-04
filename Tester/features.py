# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:33:45 2018

@author: Ashwathy 
"""
import pandas as pd
import os
import re
from preprocessor import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import RULE_KEEP
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix
class Extractor:
    
    path = r"..\corpora\bad-words.txt"
    preprocessor=PreProcessor()
    def __init__(self):
        pass
    
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
                max_features=None)
        word_vectorizer.fit(vocab_data) ##ideally use all_data (where all_data array has to be all_data = pd.concat([train_text, test_text]))
        histogram = word_vectorizer.transform(data)
        print histogram.shape
        return histogram
    
    def get_word2vec_features(self,data):
        ''' @params - data :list of comments from which to extract features
            @output - Word2Vec features
        '''
        
        split_data=self.preprocessor.split_sentences(data)
        
        '''use these lines to train and save the word2vec model'''
        print('Creating word2vec vocabulary. You should remove these lines of code if you already did this.')
        model = Word2Vec(split_data, size=100, window=5, min_count=5, workers=4, hs=1, negative =0)
        model.save(r"..\corpora\toxic.model")
        model.wv.save_word2vec_format(r"..\corpora\toxic.model.bin", binary=True)

        '''use this line when already trained a word2vec model'''
        model = Word2Vec.load(r"..\corpora\toxic.model")
        '''The juice'''
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        dim = len(next(iter (w2v.values())))
        return np.array([
            np.mean([w2v[w] for w in sentence if w in w2v]
                    or [np.zeros(dim)], axis=0)
            for sentence in data
                ])
   
    
    def build_word_vectors(self,data,vocab_data):
        ''' @params - data :list of comments from which to extract features
            @output - Word2Vec features
        '''
        vocab_data=self.preprocessor.get_sentences(vocab_data)
        data=self.preprocessor.get_sentences(data)
        model = Word2Vec(min_count=5, hs=1,window=5, negative=5)
        model.build_vocab(vocab_data)
        model.train(data, total_examples=data.size, epochs=5) 
        return model

    def get_word_vector_model(self,data,vocab_data):
        filename = r"..\corpora\w00v.model"
        if(os.path.exists(filename)):
            model = Word2Vec.load(filename)
        else:
            model=self.build_word_vectors(data,vocab_data)
            model.save(filename)
            #model.wv.save_word2vec_format(filename+".bin", binary=True)
        print "vocab size-"+str(model.corpus_count)
        return model
    
    def get_word_vectors(self,data,vocab_data):
        model = self.get_word_vector_model(data, vocab_data)
        w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        dim = len(next(iter (w2v.values())))
        feature_vectors = np.array([
            np.mean([w2v[w] for w in sentence if w in w2v]
                    or [np.zeros(dim)], axis=0)
            for sentence in data
                ])
   
        #feature_vectors = np.array([ np.mean([(model[word]) for word in comment.split() if word in model],axis=0) for comment in data])
        
        return feature_vectors
              
        
    def num_bad_words(self, data):
        ''' @params - data :list of comments from which to extract features
            @output - list of number of bad words per sentence
        '''
        bad_text=pd.read_csv(self.path,sep="\n", header=None)
        bad_text.columns=['bad']
        feature_val=[]
        print("counting bad words..")
        for sentence in tqdm(data):
            count=0
            for bw in bad_text['bad']:
                #using regular expressions, each word is seperated by blank or special chars
                if re.search('(\s|^)'+bw+'(\s|$|\.)',sentence):
                    count+=1
            feature_val.append(count)
        bad_words_feature=coo_matrix(np.asarray(feature_val).transpose()).toarray()
        bad_words_feature=np.reshape(bad_words_feature,(-1,1))
        return bad_words_feature

    
    def num_censored_words(self, data):
        ''' @params - data :list of comments from which to extract features
            @output - list of number of bad words per sentence
        '''
        feature_val=[]
        for sentence in tqdm(data):
            count=0
            for words in sentence:
                #using regular expressions, each word is seperated by blank or special chars
                if re.search('(\*)'+'*'+'(\*)',words):
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


'''t=Extractor()
ls="I will kill. you","fuck you friend, you are an idiot","HELLO world",'f*ck yo'

bob=pd.Series(ls)
lol=t.get_word2vec_features(bob)
lal=t.get_word_histogram(ls,ls)

print(t.num_bad_words(ls))
print(t.num_Upper_Case(ls))
print(t.num_censored_words(ls))
'''

