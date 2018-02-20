# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:02:23 2018

@author: Apostolis Argatzopoulos, Ashwathy T Revi
"""
import numpy as np
import pandas as pd
from pprint import pprint

'''most magic happens in this file below'''
import LR_utilities as util

from gensim.models import Word2Vec #we used this previously

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

'''Decided to use logistic regression for fun'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm




'''change this if you have the data elsewhere'''
path = 'Toxic Comment data\\'
'''load data'''
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')

'''we need this for the submission file'''
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


'''function to be run on train and test data'''
def tidyup(dataframe):
    '''Noobish text preprocessing remove all chars except a-z, A-Z and 0-9'''
    clean_sentences=[]
    for item in tqdm(dataframe):
            item = util.clean_me(item)
            clean_sentences.append(item)
    '''parsing slang words'''   
    clean_slang_free_sentences=[]
    for item in tqdm(clean_sentences):
            item=util.death_to_slang(item)
            clean_slang_free_sentences.append(item)##.split()) #split() is required to make a vector of sentences and words for word2vec
    df = pd.Series(clean_slang_free_sentences)
    return df

'''clean both data'''
train_clean=tidyup(train['comment_text'])
test_clean=tidyup(test['comment_text'])


     
'''the baseline function requires data to be merged'''       
frames = [train_clean, test_clean]
all_clean = pd.concat([train_clean, test_clean])

'''may the force be with us, here we send everything to the baseline function'''




'''freshly generated from the baseline_Me function'''
train_features=util.baseline_Me(train_clean,all_clean)
test_features=util.baseline_Me(test_clean,all_clean)

'''after we got out vectorised words we now do some logistic regression, because we rock!!'''
'''dont freak out, I copied the code from https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams '''

losses = []
predictions = {'id': test['id']}
for class_name in tqdm(class_names):
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=5, scoring='roc_auc'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))

    classifier.fit(train_features, train_target)
    predictions[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(losses)))

submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('submission.csv', index=False)










'''old word2vec things below might be of use in the future'''

'''use these lines to train and save the word2vec model'''
#model = Word2Vec(text, size=100, window=5, min_count=5, workers=4, hs=1, negative =0)
#model.save('toxic.model')
#model.wv.save_word2vec_format('toxic.model.bin', binary=True)

'''use this line when already trained a word2vec model'''
#model =Word2Vec.load('toxic.model')



'''I'm just printing line 20 here just for debug. You can add lines like''' 
#model.similar_by_word ('Greek', topn = 5)
'''or'''
#model.doesnt_match("mother father damage son".split())
'''or'''
#model.most_similar(['crazy'])
'''to play around with word2vec'''

'''here i was just printing sentence number 20'''
#pprint(text[20]) 

