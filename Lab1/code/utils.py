#!/usr/bin/env python
# coding: utf-8

# In[25]:


import json
import re
from harvesttext import HarvestText
from ltp import LTP

# In[26]:


'''字典树结构，构建停用词词典'''
class Trie:
    def __init__(self):
        self.root = {}  # 用字典存储
        self.end_of_word = '#'   # 用#标志一个单词的结束
        
    def insert(self, word: str):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[self.end_of_word] = self.end_of_word

    # 查找一个单词是否完整的存在于字典树里，判断node是否等于#
    def search(self, word: str):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_of_word in node


# In[27]:


'''基于路径文件构建停用词字典树'''
def get_stop_dic(file_path):
    stop_dic = Trie()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stop_dic.insert(line.strip())
    return stop_dic



# In[28]:

'''
构建停用词字典树
'''
stop_dic = get_stop_dic('../data/stopwords.txt')


# In[29]:


'''读写文件'''
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, save_path, indent=None): # 默认一行行存储
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=indent, ensure_ascii=False))

# 存储为一行一个json格式的page文件
def pages_to_json(pages, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for data in pages:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str + '\n')
            
# 读入pages json文件           
def read_pages(file_path):
    pages_dic = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            p = json.loads(line)
            pages_dic.append(p)
    return pages_dic


# In[30]:


'''
分词，去除停用词
'''
ht = HarvestText()
ltp = LTP()
def tokenize(content:str):
    
    # 数据清洗部分
    content = ht.clean_text(content)
    
    # 分词
    words = []
    segs,_ = ltp.seg([content])
    for w in segs[0]:
        if stop_dic.search(w) or w.strip()=='':
            continue
        else:
            words.append(w.strip())
        
    return words # 返回分词结果列表

