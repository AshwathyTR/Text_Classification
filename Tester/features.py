# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:33:45 2018

@author: Ashwathy 
"""
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

class Extractor:
    
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
    