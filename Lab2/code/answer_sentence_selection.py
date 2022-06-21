#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import save_json, read_json, show_len, tokenize, get_pos
import json
import pandas as pd
import os
import jieba
import numpy as np
import math


# # 数据预处理

# In[2]:


# 分词文档 
psg = read_json('../data/passages_seg.json')


# In[3]:


# 未分词文档
noseg = read_json('../data/passages_multi_sentences.json')


# In[4]:


# 加载train.json数据并进行格式处理
def load_data3(data_path): 
    dts = []
    fl = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            p = json.loads(line)
            sent = p['answer_sentence'][0]
            pid = p['pid']
            sents = noseg[pid]['document']
            try:
                dt = {}
                idx = sents.index(sent)
                dt['idx'] = idx
                dt['ans'] =  ' '.join(tokenize(sent))
                dt['ques'] = ' '.join(tokenize(p['question'], remove_stop=False))
                dt['sents'] = psg[pid]['document']
                dt['qid'] = p['qid']
                dts.append(dt)
            except:
#                 print(pid)
                fl.append({'ans':sent,'sents':sents })
    save_json(dts, '../result/ans_train.json')
    save_json(fl,'../result/faulse_data.json')
    return dts, fl


# In[5]:


df_train, f = load_data3('../data/train.json')


# In[7]:


df_train = pd.DataFrame(df_train, columns=['qid', 'ques','sents','ans','idx'])
df_train.head()


# # 按qid升序排列读入数据用于模型训练

# In[8]:


from operator import itemgetter


# In[9]:


# 按qid升序排列
df = read_json('../result/ans_train.json')
df_by_qid = sorted(df, key=itemgetter('qid'))
df_show = pd.DataFrame(df_by_qid, columns=['qid', 'ques','sents','ans','idx'])
df_show.head()


# In[10]:


lens = [len(dd['sents']) for dd in df_by_qid]
show_len(lens, '文章句子数')
print(np.mean(lens))


# ## BM25模型用于计算bm25分数

# In[11]:


from bm25 import BM25


# ## 构建Ranking SVM的训练测试数据

# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import norm
from distance import levenshtein as edit_dist
from tqdm import tqdm, trange


# ### 各特征计算函数

# In[13]:


# 计算实词数
def cal_posnum(sent):
    tags = {'a', 'n', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v'} # 实词的tag
    seg, pos = get_pos(sent) # 词性标注
    num = 0
    for tag in pos:
        if tag in tags:
            num += 1
    return num


# In[14]:


# 计算最长公共子序列的长度占比
def cal_LCS(ques, ans):
    ques = ques.split(' ')
    ans = ans.split(' ')
    len1, len2 = len(ques)+1, len(ans)+1  # 二维数组长宽
    dp = [[0 for i in range(len1)] for i in range(len2)]  # 初始化数组
    for i in range(1, len2):  # 行遍历
        for j in range(1, len1):  # 列遍历
            if ques[j-1] == ans[i-1]:  # 对应字符串相同的情况
                dp[i][j] = dp[i-1][j-1] +1  # 直接加一
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])  # 不同的情况取更长的一种情况
    lcs = dp[-1][-1] #右下角的值，即为最长公共子序列
    return lcs/min(len1-1,len2-1)


# In[15]:


# 计算unigram bigram词共现比例
def cal_gramratio(ques, ans):
    ques = ques.split(' ')
    ans = ans.split(' ')
    # unigram
    uni = len([word for word in ques if word in ans]) / len(ans)
    # bigram
    q_lst = [word0 + word1 for word0, word1 in zip(ques[:-1], ques[1:])]
    ans_lst = [word0 + word1 for word0, word1 in zip(ans[:-1], ans[1:])]
    bi = len([bi_word for bi_word in q_lst if bi_word in ans_lst]) / (len(ans_lst) + 1)
    return uni, bi


# In[16]:


# 计算编辑距离
def cal_editdis(ques, ans):
    return edit_dist(ques.split(' '), ans.split(' '))


# ## 词向量/余弦相似度
# ###  加入后效果并不是很好，未加入最终模型

# In[93]:


from gensim.models import word2vec, keyedvectors
from gensim.corpora.dictionary import Dictionary


# In[95]:


w2v=keyedvectors.load_word2vec_format('../data/embedding/sgns.weibo.bin')


# In[113]:


from sklearn.feature_extraction.text import CountVectorizer


# In[258]:


# 余弦相似度
def cal_cosim(ques, ans):
    tcv = CountVectorizer()
    combined = [ques, ans]
    vec_1 = tcv.fit_transform(combined).toarray()[0]
    vec_2 = tcv.fit_transform(combined).toarray()[1]
    if len(vec_1) != len(vec_2):
        return 0
    s = sum(vec_1[i] * vec_2[i] for i in range(len(vec_2)))
    den1 = math.sqrt(sum([pow(num, 2) for num in vec_1]))
    den2 = math.sqrt(sum([pow(num, 2) for num in vec_2]))
    return s / (den1 * den2) if (den1 * den2) !=0 else 0


# ## 计算所有特征并按rank_svm格式整合

# In[17]:


def get_features(df_by_qid):
    train_flist = []
    for qa_dic in tqdm(df_by_qid):
        qid = qa_dic['qid']
        sents = qa_dic['sents']
        ques = qa_dic['ques']
        # 利用sents初始化BM25模型
        bm25 = BM25([], sents)
        scores = bm25.cal_scores(ques)
        # 利用sents初始化tfidf向量
        tv = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        tv.fit_transform(' '.join(ans) for ans in sents)
        # 计算各个特征
        i = 0
        for ans in sents:
            features = []
            # BM25相似度
            features.append(f'1:{scores[i]}')
            # tfidf相似度
            vecs = tv.transform([ques, ans]).toarray()
            norm_val = (norm(vecs[0]) * norm(vecs[1]))
            tfidf_score = (np.dot(vecs[0], vecs[1]) / norm_val) if norm_val else 0
            features.append(f'2:{tfidf_score}')
            # 编辑距离
            edis = cal_editdis(ques, ans)
            features.append(f'3:{edis}')
#             # 词向量相似度
#             n_sim = w2v.n_similarity(ques.split(' '), ans.split(' '))
#             features.append(f'4:{n_sim}')
            # 候选答案句实词个数
            posnum = cal_posnum(ans)
            features.append(f'4:{posnum}')
            # unigram bigram词共现比例
            uni, bi = cal_gramratio(ques, ans)
            features.append(f'5:{uni}')
            features.append(f'6:{bi}')
            # LCS比例
            lcs = cal_LCS(ques, ans)
            features.append(f'7:{lcs}')
            
            # 根据idx为value赋值
#             value = 20 if i == qa_dic['idx'] else 0
            value = 20 if i == qa_dic['idx'] else 1
#             value = 3 if i == qa_dic['idx'] else 0
            i += 1
            # 将特征按照空格合并
            final_f = ' '.join(features)
            # 按ranking SVM格式写入train_flist训练特征集
            train_flist.append(f'{value} qid:{qid} {final_f}')
    return train_flist


# In[30]:


flist2 = get_features(df_by_qid)


# In[51]:


train_flist = get_features(df_by_qid)


# In[31]:


# 训练特征写入文件
def save_features(train_flist, path):
    with open(path ,'w', encoding='utf-8') as f:
        f.write('\n'.join(train_flist))


# In[32]:


# 分割训练测试集，并写入文件
def split_features(train_flist, test_size, tag):
    split_idx = int(len(df_by_qid)*(1-test_size))
    split_qid = int(df_by_qid[split_idx]['qid']) # 分割处的qid
    train = []
    test = []
    for train_f in tqdm(train_flist):
        qid = train_f.split()[1][4:] # 提取qid（str）
        if int(qid) <= split_qid:
            train.append(train_f)
        else:
            test.append(train_f)
    save_features(train, f'../data/features_train{tag}.dat')
    save_features(test, f'../data/features_test{tag}.dat')


# In[204]:


split_features(train_flist, 0.2, '')


# In[33]:


split_features(flist2, 0.2, 2)


# # 模型训练

# In[34]:


# 调用svm-rank可执行文件，训练并预测模型
def train_rank_svm(train_data_path, model_path):
    train_cmd = f'.\svm_rank\svm_rank_learn.exe -c 200.0 {train_data_path} {model_path}'
    os.system(train_cmd)
    
def test_rank_svm(test_data_path, model_path, pre_path):
    predict_cmd = f'.\svm_rank\svm_rank_classify.exe {test_data_path} {model_path} {pre_path}'
    os.system(predict_cmd)


# In[55]:


tag = 2
train_data_path = f'../data/features_train{tag}.dat'
test_data_path = f'../data/features_test{tag}.dat'
pre_path = f'../result/svm_predict{tag}.dat'
model_path = f'../model/ranking_svm{tag}.dat'


# In[36]:


train_rank_svm(train_data_path, model_path)
test_rank_svm(test_data_path, model_path, pre_path)


# In[60]:


def get_ans(test_data_path, pre_path):
    with open(test_data_path, 'r', encoding='utf-8') as f1:
        features = f1.readlines()
    with open(pre_path, 'r', encoding='utf-8') as f2:
        pres = f2.readlines()
        
        y_real, y_pre, right = {}, {}, 0
        for fs, pre in zip(features, pres):
            qid = int(fs.split()[1][4:]) # 提取qid
            # 按qid构建对应的ans list
            ans_real = y_real.get(qid, [])
            ans_pre = y_pre.get(qid, [])
            # 将ans的相关性按索引加入列表
            ans_real.append(int(fs.split()[0]))
            ans_pre.append(float(pre.strip()))
            y_real[qid], y_pre[qid] = ans_real, ans_pre

        idxs = []
        for qid in y_real:
            topidx = np.argsort(y_pre[qid])[::-1][0] # 按预测分数倒序排列并返回分值最高的句子索引
            idxs.append({'qid':qid, 'idx':topidx})
            if y_real[qid][topidx] == 20:
                right += 1
        print(f'测试集共计qid:{len(y_real)}个 正确数:{right}个 模型准确率:{right/len(y_real)}')
        return idxs


# In[61]:


# LCS 200 7 20-0
# test_rank_svm(test_data_path, model_path, pre_path)
idxs = get_ans(test_data_path, pre_path)


# In[62]:


# LCS 200 7 20-1
# test_rank_svm(test_data_path, model_path, pre_path)
idx2 = get_ans(test_data_path, pre_path)

