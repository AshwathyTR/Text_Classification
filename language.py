# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:37:34 2018
@author: Ashwathy T Revi

Contains methods for extracting features that invlove language processing.

"""
from gensim.models import Word2Vec
from dataset import CSV_Parser
import os.path


import re    #for regex
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer 
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer  
#contains appostrophe corrections 
import appos



class PreProcessor:
    
    parser = CSV_Parser()
    eng_stopwords = set(stopwords.words("english"))

    lem = WordNetLemmatizer()
    tokenizer=TweetTokenizer()
    APPO = appos.appos
    
    def clean(self,comment):
        """
        This function was taken from Kaggle - Stop the S@as
        This function receives comments and returns clean word-list
        """
        #Convert to lower case , so that Hi and hi are the same
        comment=comment.lower()
        #remove \n
        comment=re.sub("\\n","",comment)
        # remove leaky elements like ip,user
        comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
        #removing usernames
        comment=re.sub("\[\[.*\]","",comment)
        #removing non-ascii characters 
        #re.sub(r'[^\x00-\x7f]',r'', ' ')
        #comment = re.sub(r'[\x83,\x8f, \x8e, \xf1, \x8b, \xaf, \xa8, \xff, \x8d]',r'',comment)
        
        #Split the sentences into words
        try:
            words=self.tokenizer.tokenize(comment)
        except:
            ascii_chars = ""
            for character in comment:
                try:
                    character.encode('utf-8')
                    ascii_chars = ascii_chars + character
                except:
                    pass
            print ascii_chars
            words=self.tokenizer.tokenize(ascii_chars)
       
        
        # (')aphostophe  replacement (ie)   you're --> you are  
        # ( basic dictionary lookup : master dictionary present in a hidden block of code)
        words=[self.APPO[word] if word in self.APPO else word for word in words]
        words=[self.lem.lemmatize(word, "v") for word in words]
        words = [w for w in words if not w in self.eng_stopwords]
        
        clean_sent=" ".join(words)
        # remove any non alphanum,digit character
        #clean_sent=re.sub("\W+"," ",clean_sent)
        #clean_sent=re.sub("  "," ",clean_sent)
        return(clean_sent)
        
       
        
    def get_preprocessed_data(self):
          data=self.parser.indexed_by_class()
          clean_data ={}
          for comment_class in data.keys():
              clean_data[comment_class]=[]
              for comment in data[comment_class]:
                  clean_data[comment_class].append(self.clean(comment))
          return clean_data
        
        
class Processor:
    
    models={}
    parser = CSV_Parser()
    pre_processor = PreProcessor()
    
    def build_models(self):
        models={}
        data = self.pre_processor.get_preprocessed_data()
        for comment_class in data.keys():
            print comment_class
            models[comment_class] = Word2Vec(data[comment_class], size=100, window=5, min_count=5, workers=4, hs=1, negative =0)
            models[comment_class].save(comment_class+'.model')
        self.models=models
        return models
    
    def load_models(self):
        models={}
        classes = self.parser.get_classes()
        for comment_class in classes:
           models[comment_class]= Word2Vec.load(comment_class+'.model')
        self.models= models
        return models
    
    def are_models_on_disk(self):
        models_on_disk = True
        classes = self.parser.get_classes()
        for comment_class in classes:
            if not os.path.isfile(comment_class+".model"):
                models_on_disk = False
        return models_on_disk
    
    
    def get_models(self):
        if self.models:
            return self.models
        elif self.are_models_on_disk():
            return self.load_models()
        else:
            return self.build_models()
            
            
    def get_scores_by_class(self,text):
        models = self.get_models()
        scores={}
        classes = self.parser.get_classes()
        #print models['threat'].wv['fuck']
        
        for comment_class in classes:
            scores[comment_class] = models[comment_class].score(text, total_sentences=100)
        return scores
            
        
        
p = Processor()
pp=PreProcessor()
print p.get_scores_by_class([pp.clean('')])  
    
