# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:37:32 2018

@author: hp
"""

slang_path = r"slang_dict.txt"

def parse_slang():
    slang_dict={}
    with open(slang_path,'r') as f:
        entries = f.readlines()
    for entry in entries:
        if(not '`' in entry):
            continue
        
        word = entry.split('`')[0]
        meaning = entry.split('`')[1]
        meaning = meaning.split('|')[0]
        meaning = meaning.replace('\n','')
        slang_dict[word] = meaning
    return slang_dict
        
        
        
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

''' insert data here, either train or test and get the baseline vector (or whatever you want to call it) back'''
def baselineMe(self, data):
    word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 1),
            max_features=15000)
    word_vectorizer.fit(data)
    baseline_features = word_vectorizer.transform(data)
    return baseline_features