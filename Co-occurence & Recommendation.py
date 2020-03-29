#!/usr/bin/env python
# coding: utf-8

##### is Main Process title
### is Detail
# is Remark

import numpy as np
import pandas as pd
import logging
from scipy import sparse
import random
import re

##### Upload Data
trans = pd.read_csv('02거래정보.csv')
trans = trans.query('pd_c != "unknown"').sort_values(by = 'pd_c').reset_index()
trans = trans.drop(['index'], axis = 1)
trans = trans[trans.pd_c != "0667"]
trans = trans[trans.pd_c != "0196"]
trans = trans[trans.pd_c != "0524"]

item = pd.read_csv('04상품분류정보.csv')
item = item.dropna()

df = pd.read_csv('final.csv')
df = df.query('trans != "[]"')



#### Preprocessing
# To create a co-occuerence matrix,
# the range of clnt_id and pd_c is converted from 0 to the number of unique values.
# I can use Various ways like the count, jaccard, or lift method,
# but only count method because there is no rating information

col_user_id = 'clnt_id'
col_item_id = 'pd_c'
n_users = trans.clnt_id.nunique()
n_items = trans.pd_c.nunique()
col_rating = 'rating'



### encoding
vocab_users = {}
num_users = 0
for i in np.hstack([trans[col_user_id]]):
    if vocab_users.get(i) != None:
        continue
    vocab_users[i] = num_users
    num_users += 1
    
vocab_items = {}
num_items = 0
for i in np.hstack([trans[col_item_id]]):
    if vocab_items.get(i) != None:
        continue
    vocab_items[i] = num_items
    num_items += 1
    
encoded_users = [vocab_users[i] for i in trans[col_user_id]]
encoded_items = [vocab_items[i] for i in trans[col_item_id]]

df2 = pd.DataFrame({'clnt_id':encoded_users, col_item_id:encoded_items})
df2[col_rating] = 1



### Make co-occurence matrix
user_item_hits = sparse.coo_matrix((np.repeat(1, df2.shape[0]),
                                    (df2[col_user_id], df2[col_item_id])),shape=(n_users, n_items),).tocsr()

item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)

item_cooccurrence_count = item_cooccurrence.toarray()

i_n = 195
a = item_cooccurrence_count[i_n].copy()
b = a.argsort()[-10:]
display(b)
display(item.iloc[i_n,:])
c = item.iloc[b,]
display(c)



##### Make Top-K list
pattern = '([0-9]+)'

pc = []
for i in range(df.shape[0]):
    a = df.iloc[i,1]
    pd_c = re.findall(pattern, a[1:-1])
    for j in range(len(pd_c)):
        pd_c[j] = int(pd_c[j])
    pc.append(pd_c)

for i, pcc in enumerate(pc):
    for j in range(len(pcc)):
        if pcc[j] < 195:
            pc[i][j] = pc[i][j]-1
        elif 195 < pcc[j] < 523:
            pc[i][j] = pc[i][j]-2
        elif pcc[j] > 666:
            pc[i][j] = pc[i][j]-3

f = []
for i, p in enumerate(pc):
    c = []
    g = []
    for j in range(len(p)):
        a = item_cooccurrence_count[p[j]].copy()
        b = a.argsort()[-10:]
        c.append(list(b))

        answer = sum(c, [])
        d = []
        e = list(set(answer))

        for k in range(10):
            randomIndex = random.randrange(0,len(e))
            d.append(e[randomIndex])
            del e[randomIndex]

    f.append(d)

reco_df = pd.DataFrame([],columns=['clnt_id', 'pd_c', 'clac_nm1', 'clac_nm2', 'clac_nm3'])
for i, clnt in enumerate(df.clnt_id):
    c = []
    for h in range(10):
        c.append(clnt)
    data = pd.concat([pd.DataFrame(c, columns = ['clnt_id']),
                      item.iloc[f[i],:].reset_index().drop(['index'], axis = 1)], axis = 1, sort=False)
    reco_df = reco_df.append(data)

reco_df.to_csv('top-10.csv', index = False)