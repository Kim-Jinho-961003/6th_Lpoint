#!/usr/bin/env python
# coding: utf-8

##### is Main Process title
### is Detail
# is Remark

import pandas as pd
from selenium.webdriver import Chrome
import time
import tqdm

##### Upload Data
product = pd.read_csv('04상품분류정보.csv', encoding = 'utf-8')

real_cate = product.clac_nm3



##### Crawling(Naver PaPago)
browser = Chrome()

url = 'https://papago.naver.com/?sk=en&tk=ko&hn=0'

browser.get(url)

new_keyword = []
for i in tqdm.tqdm_notebook(real_cate):
    search = browser.find_element_by_css_selector('textarea') # Input box selection
    search.send_keys(i)
    time.sleep(3)
    find = browser.find_element_by_css_selector('div#txtTarget') # Get text in outputbox
    new_keyword.append(find.text)
    time.sleep(3)
    browser.find_element_by_css_selector('#sourceEditArea > button').click() # Clear input box
    time.sleep(2)

pd.DataFrame(new_keyword).to_csv('new_product.csv',encoding = 'utf-8', index = False, header=False)