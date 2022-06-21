#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bm25 import BM25
from utils import tokenize
import joblib
import os
import pandas as pd
import json


# In[2]:


class Retriver():
    def __init__(self, web_dt, file_dt):
        self.web_dt = web_dt
        self.file_dt = file_dt
         
    def build_model(self, web_model_path, file_model_path):
        if os.path.exists(web_model_path):
            self.web_model = joblib.load(web_model_path)
        else:
            self.web_model = BM25(self.web_dt['idx'], self.web_dt['paragraphs'])
            joblib.dump(self.web_model, web_model_path)
            
        if os.path.exists(file_model_path):
            self.file_model = joblib.load(file_model_path)
        else:
            self.file_model = BM25(self.file_dt['idx'], self.file_dt['content'])
            joblib.dump(self.file_model, file_model_path)
        
    def search_web(self, query):
        query = ' '.join(tokenize(query)) # 对query分词并且去停用词
        topk = self.web_model.get_topk(query, k=5)
#         titles = [self.web_dt['title'][i] for i in topk]
        return topk

    def search_file(self, query):
        query = ' '.join(tokenize(query)) # 对query分词并且去停用词
        topk = self.file_model.get_topk(query, k=5)
#         file_names = [self.file_dt['file_name'][i] for i in topk]
        return topk

