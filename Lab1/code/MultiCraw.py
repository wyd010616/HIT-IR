#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import time
from concurrent import futures
from bs4 import BeautifulSoup
from copy import deepcopy
from urllib.request import urlopen, urlretrieve, Request
from PageCraw import Page


# In[2]:


class Craw():
    def __init__(self):
        self.urls = set()
        self.pages = []
        self.att_cnt = 0
        self.cnt = 0
        
    def show(self):
        dic = []
        for page in self.pages:
            dic.append(page.show())
        return dic
    
        
    def multi_craw(self, att_dir, urls = []):
        executor = futures.ThreadPoolExecutor(max_workers=10)
        start = time.time()
        fs = []
        urls.extend(list(self.urls))
        for url in urls:
            # 提交任务到线程池
#             print(url)
            f = executor.submit(Page.load_url, url, att_dir)
            fs.append(f)
        # 等待这些任务全部完成
        futures.wait(fs)
        # 获取任务的结果
        for f in fs:
            if f.result() is False:
                continue
            else:
                self.cnt += 1 # 成功爬取网页个数
                rr = f.result()
                self.att_cnt += rr[1] # 带有附件的网页个数
                self.pages.append(rr[0])
        end = time.time()
        print(f'多线程爬虫耗时:{end-start}s')
        print(f'共成功爬取{self.cnt}个网页数据，其中带附件网页共计{self.att_cnt}个')
        return self.pages
        
    def bfs(self, url):
        url_root = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',url)[0] # 根路径
        headers = {
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Cookie': 'SESS0b78b3575298f2ed94ea5549d866ad3c=gCKCtOU7h4zyZGkmGp2Debo-I22w6dzCkl6UfS-y8vU; Drupal.visitor.DRUPAL_UID=36477',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36'
                }
        request = Request(url=url, headers=headers) # 加入请求头
        html = urlopen(request).read().decode('utf-8')
        soup = BeautifulSoup(html, features='html.parser')
        ret = set()
        all_href = soup.find_all('a')
        head = '\/\/.+?\/'
        for h in all_href:
            try:
                next_url = h['href']
                next_url = re.sub(url_root, '', next_url)
                next_url = re.sub(head, '', next_url)
                if next_url[0] == '/'and not self.has_chinese(next_url):
                    ret.add(url_root+next_url)
            except:
                continue
        return ret

    def has_chinese(self, string):
        for ch in string:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
    
    def multi_geturls(self, urls):
        executor = futures.ThreadPoolExecutor(max_workers=5)
        start = time.time()
        fs = []
        for url in urls:
            # 提交任务到线程池
            f = executor.submit(self.bfs, url)
            fs.append(f)
        # 等待这些任务全部完成
        futures.wait(fs)
        # 获取任务的结果
        for f in fs:
            self.urls = self.urls.union(f.result())
        end = time.time()
        print(f'多线程BFS网页url耗时：{end-start}s，共计获得{len(self.urls)}个不同网页url数据')
        urls = self.urls
        return list(urls)
    