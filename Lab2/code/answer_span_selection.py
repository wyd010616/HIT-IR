#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import tokenize, save_json, read_json, get_pos, get_ner
from operator import itemgetter
from lr import LR_b, LR_s
from tqdm import tqdm, trange
from metric import bleu1
import pandas as pd
import joblib
import re
import json
import os


# ## 数据预处理

# In[2]:


# 加载train.json数据并进行格式处理
def load_data4(data_path): 
    dts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            p = json.loads(line)
            dt = {}
            dt['qid'] = p['qid']
            dt['sent'] = ' '.join(tokenize(p['question'], remove_stop=False))
            dt['ans_sent'] = ' '.join(tokenize(p['answer_sentence'][0], remove_stop=False)) # 答案句
            dt['ans'] = ' '.join(tokenize(p['answer'], remove_stop=False))
            dts.append(dt)
    return dts


# In[3]:


df = read_json('../result/p4_train.json')
df = sorted(df, key=itemgetter('qid'))
df_show = pd.DataFrame(df, columns=['qid','ques','sent','ans'])
df_show = df_show.rename(columns={'sent':'ans_sent', 'ques':'sent'})
df_show.head()


# In[4]:


df_seg = read_json('../result/p4_train_seg.json')
df_seg = sorted(df_seg, key=itemgetter('qid'))
df_seg_show = pd.DataFrame(df_seg, columns=['qid','sent','ans_sent','ans'])
df_seg_show.head()


# # 用已有模型预测类别并整合数据

# In[5]:


lr_b = joblib.load('../model/lr_B.pkl')
lr_s = joblib.load('../model/lr_S.pkl')


# In[6]:


labelb = lr_b.predict(df_seg_show)
labels = lr_s.predict(df_seg_show)
df_show['label'] = labelb
df_show['labels'] = labels
df_show.head()


# In[129]:


# dataframe转为字典list
def pd2lst(df_show):
    df_lst = []
    for i in range(len(df_show)):
        dd = df_show.iloc[i]
        dic = {'qid':str(dd['qid']), 'sent':dd['sent'], 'ans_sent':dd['ans_sent'], 'ans':dd['ans'], 'label':dd['label'], 'labels':dd['labels']}
        df_lst.append(dic)
    return df_lst


# In[130]:


# save_json(df_lst, '../result/no_seg2.json')


# # 直接读入已处理好的数据
# ## 未分词数据+label

# In[11]:


df_lst = read_json('../result/no_seg2.json')


# In[3]:


# 统计各大类
def get_qa_list(dt):
    qa_list={'HUM':[], 'NUM':[],'TIME':[],'LOC':[],'DES':[], 'OBJ':[],'UNKNOWN':[]}
    for query in tqdm(dt):
        qa = {'q':query['sent'], 'a':query['ans_sent'], 'labels':query['labels'], 'ans':query['ans'],'qid':query['qid']}
        qa_list[query['label']].append(qa)
    return qa_list


# In[4]:


qa_list = get_qa_list(df_lst)


# In[5]:


# 计算平均bleu
def cal_bleu(pre_lst, ans_lst):
    num = len(ans_lst)
    bleu = 0
    for pre, tru in zip(pre_lst, ans_lst):
        bleu += bleu1(pre, tru)
    print(f'共计{num}个样例，平均bleu为:{bleu/num}')
    return bleu/num


# In[6]:


# 对冒号的处理函数
def drop_mh(sent):
    if '：' in sent or ':' in sent or '：' in sent:
        begin = sent.index('：') if '：' in sent else sent.index(':')  # 冒号
        aa = sent[begin+1:]
    else:
        aa = sent
    return aa.strip()

def ex_mh(sent):
    aa = ''
    begin = sent.index('：') if '：' in sent else sent.index(':')  # 冒号
    for i in range(begin + 1, len(sent)):
        if sent[i] in ['。', '！', '？']:
            break
        aa += sent[i]
    return aa


# In[7]:


# 最长公共子串及索引，但加入后效果不是很好，最终没有加入
def get_maxstr(ques, ans):
    ans0 = ans
    ques, ans = (ans,ques) if len(ques)>len(ans) else (ques,ans)
    f=[]
    for i in range(len(ques),0,-1):
        for j in range(len(ques)+1-i):
            e=ques[j:j+i]
            if e in ans:
                f.append(e)
        if f:
            break
    f1 = ''.join(f)
#     print(f1)
    try:
        idx = ans0.index(f[0])
    except:
#         print(f'{f1}:{ans}')
        idx = -1
    return f, idx


# # 对各类的规则

# In[8]:


# HUM 0.5559143590965517
def get_hum_ans(qa_list):
    ans = []
    qids = []
    qlst = []
    for qa in tqdm(qa_list):
        ques = qa['q']
        sent = qa['a']
        qlst.append(ques)
        qids.append(qa['qid'])
        lb, ls = qa['labels'].split('_') #大小标签
        if lb != 'HUM':
            ans.append(drop_mh(sent))
            continue
        ners = get_ner(sent)
        if ners['Nh']:
            ans.append('、'.join(ners['Nh']))
        elif ls == 'ORGANIZATION'and ners['Ni']:
            ans.append('、'.join(ners['Ni']))
        elif '：' in sent or ':' in sent:
            ans.append(ex_mh(sent))
        else:
            ans.append(drop_mh(sent))
    return pd.DataFrame({'qid':qids, 'question':qlst, 'answer':ans})


# In[9]:


# # LOC 0.48676425856487476
def get_loc_ans(qa_list):
    ans = []
    qids = []
    qlst = []
    for qa in tqdm(qa_list):
        ques = qa['q']
        sent = qa['a']
        qlst.append(ques)
        qids.append(qa['qid'])
        if '：' in sent or ':' in sent:
            ans.append(ex_mh(sent))
            continue
        ners = get_ner(sent)
        if ners['Ns']:
            ans.append('、'.join(ners['Ns']))
            continue
        else:
            ans.append(drop_mh(sent))
            continue
    return pd.DataFrame({'qid':qids, 'question':qlst, 'answer':ans})


# In[15]:


# NUM 0.5926360860213366
def get_num_ans(qa_list):
    ans = []
    qids = []
    qlst = []
    for qa in tqdm(qa_list):
        ques = qa['q']
        sent = qa['a']
        qlst.append(ques)
        qids.append(qa['qid'])
        if '：' in sent or ':' in sent:
            ans.append(ex_mh(sent))
            continue
        seg, pos = get_pos(sent)
        result = []
        for idx, tag in enumerate(pos):
            if tag == 'm' and idx < len(pos) - 1:
                if pos[idx + 1] == 'q':
                    result.append(seg[idx] + seg[idx + 1])
                else:
                    result.append(seg[idx])
        if len(result) > 0:
            ans.append(result[0])
        else:
            ans.append(drop_mh(sent))
    return pd.DataFrame({'qid':qids, 'question':qlst, 'answer':ans})


# In[16]:


# TIME 0.645968083728072
def get_time_ans(qa_lst):
    ans = []
    qids = []
    qlst = []
    for qa in tqdm(qa_lst):
        ques = qa['q']
        sent = qa['a']
        qlst.append(ques)
        qids.append(qa['qid'])
        lb, ls = qa['labels'].split('_') # 大小标签
        result = []
        if ls == 'YEAR': # xx年/xxxx年
            result = re.findall(r'\d{2,4}年', sent)
        elif ls == 'MONTH': # x月/xx月
            result = re.findall(r'\d{1,2}月', sent)
        elif ls == 'DAY': # x日/xx日
            result = re.findall(r'\d{1,2}日}', sent)
        elif ls == 'WEEK':
            result = re.findall(r'((周|星期|礼拜)[1-7一二三四五六日])', sent)
            result = [res[0] for res in result]
        elif ls == 'RANGE': # xxxx年到xxxx年/xxx年-xxxx年
            result = re.findall(r'\d{2,4}[年]?[-到至]\d{2,4}[年]?', sent)
        else:
            result = re.findall(r'\d{1,4}[年/-]\d{1,2}[月/-]\d{1,2}[日号]?', sent)  # 年月日
            if not result:
                result = re.findall(r'\d{1,4}[年/-]\d{1,2}月?', sent)  # 年月
            if not result:
                result = re.findall(r'\d{1,2}[月/-]\d{1,2}[日号]?', sent)  # 月日
            if not result:
                result = re.findall(r'\d{2,4}年', sent)
            if not result:
                result = re.findall(r'\d{1,2}月', sent)
                
        if len(result) > 0:
            ans.append(result[0]) # 返回结果中的第一个日期
        else:
            ans.append(drop_mh(sent))
    return pd.DataFrame({'qid':qids, 'question':qlst, 'answer':ans})


# In[17]:


# OBJ 0.48366248404525575
def get_obj_ans(qa_lst):
    ans = []
    qids = []
    qlst = []
    for qa in tqdm(qa_lst):
        ques = qa['q']
        sent = qa['a']
        qlst.append(ques)
        qids.append(qa['qid'])
        if '：' in sent or ':' in sent:
            ans.append(ex_mh(sent))
        else:
            ans.append(drop_mh(sent))
    return pd.DataFrame({'qid':qids, 'question':qlst, 'answer':ans})


# In[18]:


# DES/UNK 0.5433658533737482
# DES:0.5433658533737482
# UNK:0个样例
def get_do_ans(qa_lst):
    ans = []
    qids = []
    qlst = []
    for qa in tqdm(qa_lst):
        ques = qa['q']
        sent = qa['a']
        qlst.append(ques)
        qids.append(qa['qid'])
        if '：' in sent or ':' in sent:
            ans.append(ex_mh(sent))
        else:
            ans.append(drop_mh(sent))
    return pd.DataFrame({'qid':qids, 'question':qlst, 'answer':ans})


# In[19]:


def get_all_ans(qa_list):
    hum_ans = get_hum_ans(qa_list['HUM'])
    num_ans = get_num_ans(qa_list['NUM'])
    loc_ans = get_loc_ans(qa_list['LOC'])
    time_ans = get_time_ans(qa_list['TIME'])
    obj_ans = get_obj_ans(qa_list['OBJ'])
    des_ans = get_do_ans(qa_list['DES'])
    unk_ans = get_do_ans(qa_list['UNKNOWN'])
    all_ans = pd.concat([hum_ans, num_ans, loc_ans, time_ans, obj_ans, des_ans, unk_ans], ignore_index=True)
    return all_ans


# In[20]:


all_ans = get_all_ans(qa_list)


# In[22]:


# 抽取的答案
all_ans.head()


# In[23]:


def get_true_ans(qa_list):
    hum_ans = pd.DataFrame(qa_list['HUM'], columns=['ans'])
    num_ans = pd.DataFrame(qa_list['NUM'], columns=['ans'])
    loc_ans = pd.DataFrame(qa_list['LOC'], columns=['ans'])
    time_ans = pd.DataFrame(qa_list['TIME'], columns=['ans'])
    obj_ans = pd.DataFrame(qa_list['OBJ'], columns=['ans'])
    des_ans = pd.DataFrame(qa_list['DES'], columns=['ans'])
    unk_ans = pd.DataFrame(qa_list['UNKNOWN'], columns=['ans'])
    all_ans = pd.concat([hum_ans, num_ans, loc_ans, time_ans, obj_ans, des_ans, unk_ans], ignore_index=True)
    return all_ans


# In[24]:


real_ans = get_true_ans(qa_list)


# In[25]:


# 真实答案
real_ans.head()


# In[26]:


# 0.5512000673111742
cal_bleu(all_ans['answer'], real_ans['ans'])

