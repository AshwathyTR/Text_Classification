# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:26:18 2018

@author: Croft
"""
import pandas as pd
import matplotlib.pyplot as plt

class Validator:
    
    path='output.csv'
    data = pd.read_csv(path)
    mistake = ['identity_hate_mistake','insult_mistake', 'obscene_mistake','severe_toxic_mistake', 'threat_mistake', 'toxic_mistake']
    real = ['identity_hate_real','insult_real', 'obscene_real','severe_toxic_real', 'threat_real', 'toxic_real']
    
    def count_FP(self):
        corr=0
        total=0
        fail=0
        total_classifications=0
        mistake_vec=[]
        #total_classifications=total_classifications+self.data[mistake_class].value_counts()
        for mistake_class,real_class in zip(self.mistake,self.real):
            #total=total+self.data[real_class].value_counts()
            count=0
            for a,b in zip(self.data[real_class],self.data[mistake_class]):
                if a==1:
                    total=total+1
                    if b==0:
                        corr=corr+1
                    if b==1:
                        fail=fail+1
                if b==1:
                    count+=1
                    total_classifications=total_classifications+1
            mistake_vec.append(count)
        
        '''The following data sums all classes. So it is incorrect to say these are the number of comments'''
                  
        print('Num of Toxic comments(sum of all classes): \n%s\nNum of mistakenly classified comments(sum of all classes): \n%s'%(total,total_classifications))
        print('Num of comments that are toxic and were correctly classified as such:\n%s'%(corr))
        print('Num of comments that are toxic and were classified as being not:\n%s'%(fail))
        plt.plot(self.mistake,mistake_vec)
        plt.show()
        
vld=Validator()
vld.count_FP()
        
                        
