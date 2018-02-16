# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:37:57 2018

@author: Ashwathy T Revi
@description: csv parser functions
"""

import csv
   

class CSV_Parser:
    
    data=[]
    headers = []
    path = r"C:\Coursework\Advanced ML\datasets\train.csv\short.csv"
     
    def load_data(self):
        with open(self.path,'rb') as f:
            rows = list(csv.reader(f,delimiter=','))
        self.headers = rows.pop(0)
       
        parsed=[]
        for row in rows:
            entry={}
            for value, header in zip(row,self.headers):
                entry[header]=value
            parsed.append(entry)
        self.data = parsed
        return parsed
           
            
            
    def get_classes(self):
        if not self.headers:
            self.load_data()
        classes = self.headers[2:7]
        classes.append('clean')
        return classes
        
    def indexed_by_class(self):
        if not self.data:
            self.load_data()
        classes = self.headers[2:7]
        data_by_class={}
        for cls in classes:
            data_by_class[cls]=[]
        data_by_class['clean']=[]
        for entry in self.data:
            flag=True
            for cls in classes:
                if(entry[cls]=='1'):
                    data_by_class[cls].append(entry['comment_text'])
                    flag=False
            if(flag):
                data_by_class['clean'].append(entry['comment_text'])
        return data_by_class
    


            
        