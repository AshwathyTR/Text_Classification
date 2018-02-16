# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:37:34 2018
@author: Ashwathy T Revi

Contains methods for extracting features from raw text.

"""
from gensim.models import Word2Vec
from language import Processor

class Extractor:
    
    def get_all_features(self, text):
       features={}
       features['num_of_sentences'] = Processor.get_num_of_sentences(text)
       features['num_of_words'] = Processor.get_num_of_words(text)
       features['toxic_score'] = Processor.get_scores_by_class(text)['toxic']
       
   
       
        
    