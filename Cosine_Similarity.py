#!/usr/bin/env python
# coding: utf-8

##### is Main Process title
### is Detail
# is Remark

import pandas as pd
import tqdm
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

##### Uploading Data
online = pd.read_csv('01온라인행동정보.csv', encoding = 'utf-8')
trans = pd.read_csv('02거래정보.csv', encoding = 'utf-8')
product = pd.read_csv('04상품분류정보.csv', encoding = 'utf-8')
new_keyword = pd.read_csv('new_product.csv', header=None, encoding = 'utf-8')



##### Analysis of customers with purchase records
trans_clnt_id = list(trans.clnt_id.unique())

y_clnt = online.query('clnt_id == @trans_clnt_id').sort_values(by = ['clnt_id','sess_id','hit_seq'])

search_trans = pd.merge(
    y_clnt, trans, how = 'left'
).loc[:,['clnt_id', 'sess_id','sech_kwd','pd_c']]

y_df = search_trans.dropna(thresh=3).reset_index().iloc[:,1:]


def make_str(x): # convert int to 4-length str
    if x < 10:
        return '000' + str(x)
    elif x < 100:
        return '00' + str(x)
    elif x < 1000:
        return '0' + str(x)
    else:
        return str(x)
product['pd_c'] = product['pd_c'].apply(make_str)

p_c = product.clac_nm3.sort_values().dropna().reset_index().iloc[1:,1]
p_c2 = product.sort_values(by='clac_nm3').dropna().reset_index().iloc[1:,:]

keyword = pd.DataFrame({
    'clac_nm3': p_c.reset_index().clac_nm3,
    'kor_clac_nm3': new_keyword.iloc[:,0]
})

product = pd.merge(product, keyword, how = 'left')

y_tot = pd.merge(y_df, product, left_on = 'pd_c', right_on = 'pd_c', how = 'left')



### Merge search and purchase history data
y_clnt_id = y_tot.clnt_id.unique()

y_s = []
y_t = []
for i in tqdm.tqdm_notebook(y_clnt_id):
    a = y_tot.query('clnt_id == @i')
    y_s.append(a.sech_kwd.unique())
    y_t.append(a.kor_clac_nm3.unique())

clean_y_s = []
for i in range(0, len(y_s)):
    a=[x for x in y_s[i] if str(x) != 'nan']
    clean_y_s.append(a)

clean_y_t = []
for i in range(0, len(y_t)):
    a=[x for x in y_t[i] if str(x) != 'nan']
    clean_y_t.append(a)

y_st = []
for i in range(0, len(y_s)):
    y_st.append(list(clean_y_s[i])+list(clean_y_t[i]))

y_text = []    
for i in range(0, len(y_st)):
    a = y_st[i]
    text = ''
    for j in range(0, len(a)):
        text = text + a[j] + ' '
    y_text.append(text)

    
    
### Train Word2Vec
wv = Word2Vec(size=20,
              min_count=1,
              workers=4,
              sentences=[simple_preprocess(p) for p in y_text]
             )

wv.wv['커피']

wv.wv.most_similar('커피')

wv_words = sorted(set(wv.wv.vocab))

wvmat = wv.wv[wv_words]



### Create a representative vector for each user
pre_y_text = [simple_preprocess(p) for p in y_text];pre_y_text

# Calculate the average of the coordinates of the searched and purchased keywords
y_wv = []
for i in range(0, len(pre_y_text)):
    a =  pre_y_text[i]
    b = np.zeros(20)
    for j in range(0, len(a)):
        b = b + wv.wv[a[j]]
        c = b/len(a)
    y_wv.append(c)



##### Analysis of customers without purchase records(Same Process)
n_clnt = online.query('clnt_id != @trans_clnt_id').sort_values(by = ['clnt_id','sess_id','hit_seq']).loc[:,['clnt_id', 'sess_id','sech_kwd']]

n_df = n_clnt.dropna()

# Remove users who search for words that are not in the vocabluary of trained W2V
n_df = n_df.query('sech_kwd in @wv_words')

n_clnt_id = n_df.clnt_id.unique()

n_s = []
for i in tqdm.tqdm_notebook(n_clnt_id):
    a = n_df.query('clnt_id == @i')
    n_s.append(a.sech_kwd.unique())

n_text = []    
for i in range(0, len(n_s)):
    a = n_s[i]
    text = ''
    for j in range(0, len(a)):
        text = text + a[j] + ' '
    n_text.append(text)

pre_n_text = [simple_preprocess(p) for p in n_text];pre_n_text

n_wv= []
for i in range(0, len(pre_n_text)):
    a =  pre_n_text[i]
    b = np.zeros(20)
    for j in range(0, len(a)):
        b = b + wv.wv[a[j]]
        c = b/len(a)
    n_wv.append(c)


##### Calculate cosine similarity and extract purchase history of the most similar customer
new_product = pd.concat([p_c2.reset_index(), keyword],axis=1)

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

final = pd.DataFrame({
    'clnt_id': n_df.clnt_id.unique(),
    'trans': index_t
    'sim_id': index_c
})

final.to_csv('final.csv', encoding = 'utf-8', index = False)