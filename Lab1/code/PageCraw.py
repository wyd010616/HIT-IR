#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
from bs4 import BeautifulSoup
from copy import deepcopy
from urllib.request import urlopen, urlretrieve, Request


# In[2]:


'''爬取单个网页数据'''
class Page():
    def __init__(self, url, title, paragraphs, file_name):
        self.__data = {}
        self.__data['url'] = url
        self.__data['title'] = title
        self.__data['paragraphs'] = paragraphs
        self.__data['file_name'] = file_name
#         self.__data['file_path'] = file_path

    @staticmethod
    def load_url(url, file_dir, attachment_type=('txt', 'doc', 'docx', 'xlsx')):
        att_cnt = 0
        # 爬虫时会遇到页面跳转的问题，携带请求头部进行数据爬取
        headers = {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Cookie': 'SESS0b78b3575298f2ed94ea5549d866ad3c=gCKCtOU7h4zyZGkmGp2Debo-I22w6dzCkl6UfS-y8vU; Drupal.visitor.DRUPAL_UID=36477',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36'
            }
        request = Request(url=url, headers=headers) # 加入请求头
        
        # 异常处理
        try:
            html = urlopen(request,timeout=5).read().decode('utf-8').replace(
                u'\u2003', u'').replace(u'\u2002',u'').replace(u'\xa0',u'').replace(u'\u3000',u'') # 处理decode编码的符号
        except:
            print(f'url:{url}响应异常')
            return False
        
        soup = BeautifulSoup(html, features='html.parser')

        # 提取标题
        title = soup.find_all('div', {"class": "article-title text-center"})
        if len(title)!=0:
            title = title[0].get_text().strip()
            title = title.split(' ')[0].split('|')[-1].split('/')[-1].strip()
        else:
            title = str(soup.title.string)
#             print(title)
            title = title.split(' ')[0].split('|')[0].split('/')[-1].strip()
        
        file_path = f'{file_dir}{title}' # 附件存储地址
        
        # 提取正文，没有则用标题充当正文内容
        content = soup.find_all('p')
        if content is None:
            paragraphs = title
        else:
            paragraphs = ''
            for i in range(len(content)):
                paragraphs = paragraphs + content[i].get_text()

        # 提取附件名称并下载附件
        all_href = soup.find_all('span', {'class': 'file--x-office-document'})
        file_name = []
        download_url = [] # 记录附件下载地址
        for h in all_href:
            h = h.select('a')
            cur_name = h[0].get_text() # 提取附件名称
            if cur_name.endswith(attachment_type): # 只提取指定类型的附件
                cur_url = h[0].get('href')
                download_url.append(cur_url)
                file_name.append(cur_name) # 添加附件名称至当前cur_url对应的附件名称列表

        if len(file_name) == 0:
            file_path = None
        else: # 附件列表不为空，下载附件
            att_cnt += 1
            if not (os.path.exists(file_path)):
                os.mkdir(file_path)
            for i in range(len(file_name)):
                urlretrieve(download_url[i], f'{file_path}/{file_name[i]}')
                
#         ret = Page(url, title, paragraphs, file_name, file_path)
        ret = Page(url, title, paragraphs, file_name)
        
        return (ret, att_cnt) # 返回json格式的网页数据与附件标签

    def show(self):
        dic = self.__data
        return dic

    # 重写比较方法
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__hash__ == other.__hash__
        else:
            return False
    
    def __hash__(self):
        return hash(self.__data['url'])

