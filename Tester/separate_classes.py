# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 14:25:31 2018

@author: Ashwathy
"""

from main import Framework 
from tqdm import tqdm

f = Framework()


for comment_class in tqdm(f.classes):
   
    positive = f.data.loc[f.data[comment_class] == 1]
    negative = f.data.loc[f.data[comment_class] == 0]
    
    positive.to_csv(comment_class+'.csv', index=False)
    negative.to_csv("not_"+comment_class+".csv", index=False)
    


    
    
        
    
        

