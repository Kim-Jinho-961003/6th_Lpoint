#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import tqdm
import numpy as np

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


online = pd.read_csv('01온라인행동정보.csv', encoding = 'utf-8')
trans = pd.read_csv('02거래정보.csv', encoding = 'utf-8')
product = pd.read_csv('04상품분류정보.csv', encoding = 'utf-8')

new_keyword = pd.read_csv('new_product.csv', header=None, encoding = 'utf-8')


# ### 구매기록이 있는 고객

# In[4]:


trans_clnt_id = list(trans.clnt_id.unique())


# In[5]:


y_clnt = online.query('clnt_id == @trans_clnt_id').sort_values(by = ['clnt_id','sess_id','hit_seq'])


# In[6]:


search_trans = pd.merge(
    y_clnt, trans, how = 'left'
).loc[:,['clnt_id', 'sess_id','sech_kwd','pd_c']]


# In[7]:


# 검색, 구매 둘다 NaN인 행 제거
y_df = search_trans.dropna(thresh=3).reset_index().iloc[:,1:];y_df.head()


# In[8]:


product.head()


# In[9]:


def make_str(x):
    if x < 10:
        return '000' + str(x)
    elif x < 100:
        return '00' + str(x)
    elif x < 1000:
        return '0' + str(x)
    else:
        return str(x)
product['pd_c'] = product['pd_c'].apply(make_str)


# In[10]:


p_c = product.clac_nm3.sort_values().dropna().reset_index().iloc[1:,1]
p_c2 = product.sort_values(by='clac_nm3').dropna().reset_index().iloc[1:,:]


# In[11]:


keyword = pd.DataFrame({
    'clac_nm3': p_c.reset_index().clac_nm3,
    'kor_clac_nm3': new_keyword.iloc[:,0]
})


# In[12]:


keyword


# In[13]:


product = pd.merge(product, keyword, how = 'left');product


# In[14]:


y_tot = pd.merge(y_df, product, left_on = 'pd_c', right_on = 'pd_c', how = 'left');y_tot


# #### 검색기록과 구매 카테고리 병합

# In[15]:


y_clnt_id = y_tot.clnt_id.unique()


# In[16]:


y_s = []
y_t = []
for i in tqdm.tqdm_notebook(y_clnt_id):
    a = y_tot.query('clnt_id == @i')
    y_s.append(a.sech_kwd.unique())
    y_t.append(a.kor_clac_nm3.unique())


# In[17]:


y_s[0]


# In[18]:


y_t[0]


# In[19]:


clean_y_s = []
for i in range(0, len(y_s)):
    a=[x for x in y_s[i] if str(x) != 'nan']
    clean_y_s.append(a)


# In[20]:


clean_y_t = []
for i in range(0, len(y_t)):
    a=[x for x in y_t[i] if str(x) != 'nan']
    clean_y_t.append(a)


# In[21]:


y_st = []
for i in range(0, len(y_s)):
    y_st.append(list(clean_y_s[i])+list(clean_y_t[i]))


# In[22]:


y_text = []    
for i in range(0, len(y_st)):
    a = y_st[i]
    text = ''
    for j in range(0, len(a)):
        text = text + a[j] + ' '
    y_text.append(text)


# In[23]:


y_text


# ### Word2Vec 학습

# In[24]:


wv = Word2Vec(size=20,
              min_count=1,
              workers=4,
              sentences=[simple_preprocess(p) for p in y_text]
             )


# In[25]:


wv.wv['커피']


# In[26]:


wv.wv.most_similar('커피')


# In[27]:


wv_words = sorted(set(wv.wv.vocab))


# In[28]:


wvmat = wv.wv[wv_words]


# In[29]:


wvmat.shape


# #### 각 유저에 대한 대표 벡터 생성

# In[30]:


pre_y_text = [simple_preprocess(p) for p in y_text];pre_y_text


# 검색과 구매한 키워드의 좌표를 평균내어 계산

# In[31]:


y_wv = []
for i in range(0, len(pre_y_text)):
    a =  pre_y_text[i]
    b = np.zeros(20)
    for j in range(0, len(a)):
        b = b + wv.wv[a[j]]
        c = b/len(a)
    y_wv.append(c)


# In[32]:


display(len(y_wv))
display(y_wv[0].shape)


# ### 구매없이 검색만 진행한 고객(동일 과정 진행)

# In[33]:


n_clnt = online.query('clnt_id != @trans_clnt_id').sort_values(by = ['clnt_id','sess_id','hit_seq']).loc[:,['clnt_id', 'sess_id','sech_kwd']]


# In[34]:


n_df = n_clnt.dropna();n_df


# 학습된 W2V의 voca에 없는 단어를 검색한 유저 제거

# In[35]:


n_df = n_df.query('sech_kwd in @wv_words')


# In[36]:


n_clnt_id = n_df.clnt_id.unique()


# In[37]:


n_s = []
for i in tqdm.tqdm_notebook(n_clnt_id):
    a = n_df.query('clnt_id == @i')
    n_s.append(a.sech_kwd.unique())


# In[38]:


n_text = []    
for i in range(0, len(n_s)):
    a = n_s[i]
    text = ''
    for j in range(0, len(a)):
        text = text + a[j] + ' '
    n_text.append(text)


# In[39]:


n_text


# In[40]:


len(n_text)


# In[41]:


pre_n_text = [simple_preprocess(p) for p in n_text];pre_n_text


# In[42]:


len(pre_n_text)


# In[43]:


n_wv= []
for i in range(0, len(pre_n_text)):
    a =  pre_n_text[i]
    b = np.zeros(20)
    for j in range(0, len(a)):
        b = b + wv.wv[a[j]]
        c = b/len(a)
    n_wv.append(c)


# In[44]:


display(len(y_wv))
display(y_wv[0].shape)
display(len(n_wv))
display(n_wv[0].shape)


# In[45]:


new_product = pd.concat([p_c2.reset_index(), keyword],axis=1);new_product


# ### 코사인 유사도 계산 및 가장 유사한 고객의 구매내역 추출

# In[ ]:


index_t = []
index_c = []
for i in tqdm.tqdm_notebook(range(0, len(n_wv))):
    a = i
    cos = []
    for j in range(0, len(y_wv)):
        cos.append(cosine_similarity(n_wv[a].reshape(1,-1),y_wv[j].reshape(1,-1)))
        
    cos_df = pd.DataFrame({
        'cos_sim': cos
    }).sort_values(by='cos_sim', ascending = False).reset_index()
    
    sim_c = cos_df.iloc[1,0]
    index_c.append(sim_c)
    
    b = clean_y_t[sim_c]
    sim_t = new_product.query('kor_clac_nm3 in @b').iloc[:,1].values
    index_t.append(sim_t)


# In[ ]:


final = pd.DataFrame({
    'clnt_id': n_df.clnt_id.unique(),
    'trans': index_t
    'sim_id': index_c
})


# In[ ]:


final.to_csv('final.csv', encoding = 'utf-8', index = False)

