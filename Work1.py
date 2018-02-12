import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from pprint import pprint

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from gensim.models import Word2Vec
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import sparse

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


'''change this if you have the data elsewhere'''
path = 'Toxic Comment data\\'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')

'''Noobish text preprocessing should use BeautifulSoup library or a better parser'''
text=[]
for item in train['comment_text']:
    for x in ['"', ':', '=',':::','|',']','\n','-']:
        item=item.replace(x, '')
    text.append(item.split())
 
  
            


model = Word2Vec(text, size=100, window=5, min_count=5, workers=4, hs=1, negative =0)
model.save('toxic.model')
model.wv.save_word2vec_format('toxic.model.bin', binary=True)

'''use this line when already trained a model'''
#model1 =gensim.models.KeyedVectors.load_word2vec_format('toxic.model.bin', binary=True)



'''I'm just printing line 20 here just for debug. You can add lines like 
model.similar_by_word ('Greek', topn = 5)
or
model.doesnt_match("mother father damage son".split())
or
model.most_similar(['crazy'])
to play around with word2vec
'''
pprint(text[20])

