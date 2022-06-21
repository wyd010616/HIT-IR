#!/usr/bin/env python
# coding: utf-8

# In[2]:


from utils import pages_to_json, tokenize, read_pages
from tqdm import trange


# In[3]:


if __name__ == '__main__':
    # 读入按行存储的json格式的pages数据
    pages_dic = read_pages('../data/result/craw.json')
    
    # 进行分词及去停用词处理，并用新数据覆盖原始数据
    for i in trange(len(pages_dic)):
        pages_dic[i]['segmented_title'] = tokenize(pages_dic[i].pop('title'))
        pages_dic[i]['segmented_paragraphs'] = tokenize(pages_dic[i].pop('paragraphs'))
        pages_dic[i]['file_name'] = pages_dic[i].pop('file_name') # 不改变数据格式顺序
        
    # 将处理好的数据前10行存储至指定目录下
    pages_to_json(pages_dic[:10], './data/result/preprocessed.json')

