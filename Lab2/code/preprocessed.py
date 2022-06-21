#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import read_json, save_json, stop_dic, tokenizeltp
import pandas as pd
import math
import numpy as np
import jieba
from tqdm import trange, tqdm
import joblib
from bm25 import BM25


# # 读入分词的文章数据
# ## 合并句子列表用于bm25模型训练

# In[2]:


dt = read_json('../result/passages_seg.json')
dt = pd.DataFrame(dt, columns=['pid', 'document'])
dt['document'] = [' '.join(sents) for sents in dt['document']] # 合并用于文档检索
dt.head()


# In[3]:


# 训练模型
bm25 = BM25(dt['pid'], dt['document'])


# In[4]:


# 保存模型
joblib.dump(bm25, '../model/bm25.pkl')


# ## 测试模型检索准确率

# In[4]:


train = read_json('../data/train.json')
train = pd.DataFrame(train, columns=['question','pid'])
train.head()


# In[5]:


def test_search(dt, bm25, k):
    pid_pre = []
    pid_real = dt['pid']
    for ques in tqdm(dt['question']):
        res = bm25.get_topk(' '.join(jieba.cut(ques)), k)
        pid_pre.append(res)
        
    r_num = 0
    num = len(pid_real)
    for i in range(num):
        if pid_real[i] in pid_pre[i]:
            r_num += 1
    r = r_num/ num
    f = 2 * r * r / (r + r)
    print(f'r:{r}  f:{f}')
    return pid_pre


# In[10]:


# jieba+bm25检索效果 top1
test_search(train, bm25, 1)


# In[32]:


# jieba+bm25检索效果 top3
test_search(train, bm25, 3)


# In[8]:


def test_searchltp(dt, bm25):
    pid_pre = []
    pid_real = dt['pid']
    for ques in tqdm(dt['question']):
        res = bm25.get_topk(' '.join(tokenizeltp(ques)))
        pid_pre.append(res)
        
    r_num = 0
    num = len(pid_real)
    for i in range(num):
        if pid_real[i] in pid_pre[i]:
            r_num += 1
    r = r_num/ num
    f = 2 * r * r / (r + r)
    print(f'r:{r}  f:{f}')
    return pid_pre


# In[16]:


# ltp+bm25检索效果 top3
test_searchltp(train, bm25)


# # 对test.json进行检索

# In[9]:


test = read_json('../data/test.json')
test = pd.DataFrame(test, columns=['qid','question'])
test.head()


# In[10]:


def get_top13(dt, bm25):
    pid_pre = []
    qlst = []
    qids = []
    topids = []
    for qid, ques in tqdm(zip(dt['qid'], dt['question'])):
        res = bm25.get_topk(' '.join(jieba.cut(ques)), 3) #去停用词处理
        pid_pre.append(res)
        topids.append(res[0])
        qlst.append(ques)
        qids.append(qid)
    return pd.DataFrame({'qid':qids, 'question':qlst, 'answer_pid':pid_pre, 'pid':topids})


# In[11]:


# 排名第一的pid
pids = get_top13(test, bm25)


# In[12]:


pids.head()


# In[13]:


# dataframe转为字典list
def pd2lst(df_show):
    df_lst = []
    for i in range(len(df_show)):
        dd = df_show.iloc[i]
        anspid = [int(pp) for pp in dd['answer_pid']]
        dic = {'qid':int(dd['qid']), 'question':dd['question'], 'answer_pid':anspid, 'pid':int(dd['pid'])}
        df_lst.append(dic)
    return df_lst


# In[14]:


part1 = pd2lst(pids)


# In[15]:


save_json(part1, '../result/part1.json')

