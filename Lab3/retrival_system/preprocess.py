#!/usr/bin/env python
# coding: utf-8

# In[2]:


from utils import read_json, save_json, tokenize
from tqdm import trange, tqdm
from harvesttext import HarvestText
from win32com.client import Dispatch
import docx
import pandas as pd
import json
import joblib
import os


# In[3]:


data_web = read_json('../data/craw.json')
# file_path: title/file_name


# In[4]:


# 前1000个网页数据
# data = data_web[:1000]
web_dt = pd.DataFrame(data_web[:1000], columns=['url', 'title', 'paragraphs', 'file_name'])
web_dt['idx'] = [i for i in range(len(web_dt))]
web_dt.head()


# # 处理附件，读取文件内容用于构建检索模型

# In[5]:


# 将doc与docx一起处理
def process_doc(file_name):
    pps = []
    doc = word.Documents.Open(FileName=f'{file_dir}{file_name}', Encoding='utf-8-sig')
    try:
        for para in doc.paragraphs:
            pp = ht.clean_text(para.Range.Text)
            pp = repr(pp).replace('\\x07', '')[1:-1].replace('\\r', '').replace(' ','')
            if pp:
                pps.append(pp)
    except:
        print('erro')
    doc.Close()

    content = ''.join(pps)
#     seg = ' '.join(tokenize(content))
    return content


# In[6]:


file_dir = 'G:\\jupyter-notebook\\信息检索\\lab3-wyd\\data\\attachment\\' # 全局绝对路径
ht = HarvestText() # 用于数据清洗


# In[7]:


word = Dispatch('Word.Application')  # 全局word应用程序
# word = DispatchEx('Word.Application') # 启动独立的进程
word.Visible = 0  # 后台运行,不显示
word.DisplayAlerts = 0  # 不警告


# In[21]:


# 处理文档内容，获得分词数据
file_dt = []
for i in trange(len(dt)):
    title = dt['title'][i]
    files = dt['file_name'][i]
    if not files:
        continue
    for fn in files:
        file_name = f'{title}\\{fn}'
        content = process_doc(file_name)
        file_dt.append({'file_name':fn, 'content':content})


# In[8]:


word.Quit()


# In[9]:


save_json(file_dt, '../data/files_2.json')


# In[10]:


file_dt = read_json('../data/files_2.json')
file_dt = pd.DataFrame(file_dt, columns=['file_name', 'content'])
file_dt['idx'] = [i for i in range(len(file_dt))]
file_dt.head()


# # 构建网页附件检索模型

# In[12]:


from RetriverModel import Retriver


# In[13]:


retriver = Retriver(web_dt, file_dt)


# In[14]:


web_model_path = '../model/web_model.pkl'
file_model_path = '../model/file_model.pkl'
retriver.build_model(web_model_path, file_model_path)


# In[19]:


# 保存检索模型
joblib.dump(retriver,'../model/retriver_model.pkl')


# In[18]:


# 获得top5数据的索引
retriver.search_web('招聘专职辅导员')


# In[17]:


retriver.search_file('研究生思想政治工作')

