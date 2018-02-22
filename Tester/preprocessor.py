from gensim.models import Word2Vec
import os.path


import re    #for regex
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer 
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer  
#contains appostrophe corrections
import sys
lib_path = r'..\corpora'
sys.path.insert(0, lib_path)
import appos
path = r"..\Toxic Comment Data\less_slang.csv"
slang = pd.read_csv(path)
from tqdm import tqdm

class PreProcessor:
    
    eng_stopwords = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    tokenizer=TweetTokenizer()
    APPO = appos.appos
    slang_dict={}
    
    slang_path = os.path.join(lib_path,r"slang_dict.txt")
    
    def __init__(self):
        self.slang_dict =self.parse_slang()

   
    def parse_slang(self):
        #function to get slang_dict from slang_dict.txt
        slang_dict={}
        with open(self.slang_path,'r') as f:
            entries = f.readlines()
        for entry in entries:
            if(not '`' in entry):
                continue
            
            word = entry.split('`')[0]
            meaning = entry.split('`')[1]
            meaning = meaning.split('|')[0]
            meaning = meaning.replace('\n','')
            slang_dict[word.lower()] = meaning
        return slang_dict

    def clean(self,comment):
        """
        This function was taken from Kaggle - Stop the S@as
        This function receives comments and returns clean word-list
        """
        comment=comment.lower()
        #remove \n
        comment=re.sub("\\n","",comment)
        # remove leaky elements like ip,user
        comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
        #removing usernames
        comment=re.sub("\[\[.*\]","",comment)
        #removing non-alphabet characters 
        comment = self.remove_non_alpha(comment)
        #Split the sentences into words 
        words=self.tokenizer.tokenize(comment)
        # (')aphostophe  replacement (ie)   you're --> you are  
        words=[self.APPO[word] if word in self.APPO else word for word in words]
        words=[self.lem.lemmatize(word, "v") for word in words]
        #remove stop words
        words = [w for w in words if not w in self.eng_stopwords]
        
        clean_sent=" ".join(words)
        return(clean_sent)

    def remove_non_alpha(self,text):
        alpha = ""
        for char in text:
            if (char in range('a','z') or char in [' ','\n']):
                alpha = alpha+char
        return alpha
                
    def remove_non_ascii(self,text):
            ascii_chars = ""
            for character in text:
                try:
                    character.encode('utf-8')
                    ascii_chars = ascii_chars + character
                except:
                    pass
            return ascii_chars
            
        
        
    def clean_data(self,dataframe):
        '''@params = dataframe: the dataframe['comment_text']
           @output - a panda series with all the clean sentences
        '''
        clean_sentences=[]
        for item in tqdm(dataframe):
            item = self.clean_me(item)
            clean_sentences.append(item)
        '''parsing slang words'''   
        clean_slang_free_sentences=[]
        for item in tqdm(clean_sentences):
            item=self.remove_slang(item)
            clean_slang_free_sentences.append(item)##.split()) #split() is required to make a vector of sentences and words for word2vec
        df = pd.Series(clean_slang_free_sentences)
        return df
        
        
        
    def clean_me(self,comment):
            comment=re.sub("[^a-zA-Z0-9]", " ", comment)
            comment = comment.upper()
            return(comment)
    
    def remove_slang(self,text):
        '''@params - text: comment, sentence
           @output - cleaned comment
        '''
        comment=text
        for index,row in enumerate(slang.itertuples(),1):
            comment=comment.replace(str(row.slang),str(row.meaning))
        return(comment)
                
            

