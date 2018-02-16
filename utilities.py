# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:37:32 2018

@author: hp
"""

slang_path = r"C:\Coursework\Advanced ML\slang_dict.txt"

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
        
        
        
