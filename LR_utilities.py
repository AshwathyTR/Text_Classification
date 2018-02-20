# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:02:23 2018

@author: Apostolis Argatzopoulos, Ashwathy T Revi
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re 
path = 'Toxic Comment data\\'
slang = pd.read_csv(path+'less_slang.csv')

 
'''function that turns our words into a vectorized space... at least we hope'''
def baseline_Me(tovec_text,all_text):
    word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 1),
            max_features=15000)
    word_vectorizer.fit(all_text)
    word_features = word_vectorizer.transform(tovec_text)
    return word_features
    
'''funky function that removes all the mumbo-jumbo characters'''
def clean_me(comment):
    comment=re.sub("[^a-zA-Z0-9]", " ", comment)
    comment = comment.upper()
    return(comment)
    
'''the name says it all, replaces 'wtf' with 'what the fuck' etc.'''
def death_to_slang(comment):
    for index,row in enumerate(slang.itertuples(),1):
        #comment = re.compile(re.escape(comment), re.IGNORECASE)
        comment=comment.replace(str(row.slang),str(row.meaning))#don't try to understand the logic... it has none
        #comment = comment.lower()
    return(comment)

'''comm='WTF dude lol'
comm=clean_me(comm)
print(death_to_slang(comm))''' 