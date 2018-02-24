# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:26:18 2018

@author: Croft
"""
import pandas as pd


class Validator:
    
    path='output.csv'
    data = pd.read_csv(path)
    mistake = ['identity_hate_mistake','insult_mistake', 'obscene_mistake','severe_toxic_mistake', 'threat_mistake', 'toxic_mistake']
    real = ['identity_hate_real','insult_real', 'obscene_real','severe_toxic_real', 'threat_real', 'toxic_real']
    
    def count_FP(self):
        corr=0
        total=0
        total_classifications=0
        for mistake_class in self.real:
            total_classifications=total_classifications+self.data[mistake_class].value_counts()
            for real_class in self.mistake:
                total=total+self.data[real_class].value_counts()
                for a,b in zip(self.data[real_class],self.data[mistake_class]):
                    if a==1:
                        if b==0:
                            corr=corr+1
        '''The following data sums all classes. So it is incorrect to say these are the number of comments'''               
        print('Num of non-Toxic/Toxic comments(sum of all classes): \n%s\n\nNum of Correctly/mistakenly classified comments(sum of all classes): \n%s'%(total,total_classifications))
        print('\nNum of comments that are toxic and were correctly classified as such:\n%s'%(corr))
    
vld=Validator()
vld.count_FP()
        
                        
