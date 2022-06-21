#!/usr/bin/env python
# coding: utf-8

# In[1]:


from MultiCraw import Craw
from utils import pages_to_json


# In[2]:


if __name__ == '__main__':
    att_dir = '../data/attachment/' # 附件存储路径
    save_dir = '../data/result/' # 结果存储路径
    
    # bfs 根url获取多个url（线程池默认最大工作数目为5）
    uus = [f'http://today.hit.edu.cn/category/10?page={i}' for i in range(50)]
    mm = Craw()
    urls = mm.multi_geturls(uus)
    
    # 多线程爬取urls的网页数据，存储在指定路径文件下（线程池默认最大工作数目为10）
    pages = mm.multi_craw(att_dir)
    pages_dic = mm.show()
    pages_to_json(pages_dic, f'{save_dir}craw.json')

