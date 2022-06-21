#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import save_json, read_json, tokenizeltp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import os
import jieba
import numpy as np
import joblib
# import lightgbm as lgb


# In[2]:


# label2num = {'HUM':0, 'LOC':1, 'NUM':2, 'TIME':3, 'OBJ':4, 'DES':5, 'UNKNOWN':6}


# In[3]:


def load_data(data_path):  # 加载问题分类数据
    dt = {'label':[],'Blabel':[],'sent':[]}
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            [label, sent] = line.strip().split('\t')
            dt['sent'].append(' '.join(tokenize(sent, remove_stop=False)))
            dt['label'].append(label)
            dt['Blabel'].append(label.split('_')[0])
#             dt['Blabel'].append(label2num[label.split('_')[0]])
    dt['id'] = [i for i in range(len(dt['sent']))]
    return dt


# In[5]:


df_train = load_data('../data/trian_questions.txt')


# In[3]:


df_train = pd.DataFrame(df_train, columns=['id','label','Blabel', 'sent'])
print(f'数据集大小：{len(df_train)}')
df_train.head()


# In[2]:


# 读入测试数据转为DataFrame格式用于模型预测
df_test = load_data('../data/test_questions.txt')
df_test = pd.DataFrame(df_test, columns=['id','label','Blabel', 'sent'])
print(f'数据集大小：{len(df_test)}')
df_test.head()


# # LightGBM模型

# In[8]:


class LGBM_tfidf():        
    def _prepare(self, df_train):
        x_train, x_test, y_train, y_test = train_test_split(df_train['sent'], df_train['Blabel'], test_size=0.2)
        self.tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        train_data = self.tv.fit_transform(x_train)
        test_data = self.tv.transform(x_test)
        dtrain = lgb.Dataset(train_data, label=y_train)
        dvalid = lgb.Dataset(test_data, label=y_test)
        return dtrain, dvalid
    
    def fit(self, df_train):
        dtrain, dvalid = self._prepare(df_train)
        
        params = {'num_leaves': 50,
          'min_data_in_leaf': 10,
          'objective': 'multiclass',
          'num_class': 7,
          'max_depth': 5,
          'learning_rate': 0.05,
          "min_sum_hessian_in_leaf": 5,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          'metric': 'multi_logloss',
          # 'device': 'gpu' 
          }
        
        self.model = lgb.train(params,
                        dtrain,
                        num_boost_round=1000,
                        valid_sets=[dtrain, dvalid],
                        early_stopping_rounds=100,
                        verbose_eval=True,
                        )
        
    def predict(self, df_test):
        class_pred = []
        sents = self.tv.transform(df_test['sent'])
        class_pred = self.model.predict(sents)
        class_pred = [list(x).index(max(x)) for x in class_pred]
        return class_pred


# In[84]:


tfidflgb = LGBM_tfidf()
tfidflgb.fit(df_train)


# In[85]:


y_pre = tfidflgb.predict(df_test)


# In[86]:


#lightgbm
print(classification_report(df_test['Blabel'], y_pre))


# # LR 模型
# ## 大类

# In[5]:


class LR_b():
    def _prepare(self, df_train):
        self.tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        train_data = self.tv.fit_transform(df_train['sent'])
        return train_data
    
    def fit(self, df_train):
        x_train = self._prepare(df_train)
        y_train = df_train['Blabel']
        lr = LogisticRegression(max_iter=400, n_jobs=-1)
        param_grid = [{'C': [100, 500, 1000, 5000]}]
        self.model = GridSearchCV(lr, param_grid, cv=3, n_jobs=-1).fit(x_train, y_train)
        
    def predict(self, df_test):
        x_test = self.tv.transform(df_test['sent'])
        y_pre = self.model.predict(x_test)
        return y_pre


# In[6]:


lr = LR_b()
lr.fit(df_train)


# In[10]:


# 保存模型
dirs = '../model/lr_B.pkl'
joblib.dump(lr, dirs)


# In[7]:


# precision:0.91
y_pre_s = lr.predict(df_test)
print(classification_report(df_test['Blabel'], y_pre_s))


# ## 小类

# In[4]:


# 小类
class LR_s():
    def _prepare(self, df_train):
        self.tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        train_data = self.tv.fit_transform(df_train['sent'])
        return train_data
    
    def fit(self, df_train):
        x_train = self._prepare(df_train)
        y_train = df_train['label']
        lr = LogisticRegression(C=500)
        self.model = lr.fit(x_train, y_train)
        
    def predict(self, df_test):
        x_test = self.tv.transform(df_test['sent'])
        y_pre = self.model.predict(x_test)
        return y_pre


# In[6]:


lr_s = LR_s()
lr_s.fit(df_train)


# In[7]:


joblib.dump(lr_s,'../model/lr_S.pkl')


# In[8]:


# precision:0.7826780070573411
y_pre_lr = lr_s.predict(df_test)
print(classification_report(df_test['label'], y_pre_lr))


# In[48]:


f1_score(df_test['label'], y_pre_lr, average='weighted')

