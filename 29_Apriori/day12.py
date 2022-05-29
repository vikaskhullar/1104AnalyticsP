# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:41:19 2022

@author: vikas
"""

#Case Study Market Basket Analysi

import pandas as pd
import numpy as np


df = pd.read_csv('store_data1.csv', header=None)
df

df.shape

rec = []

for i in range(0, len(df)):
    print(i)
    rec.append([str(df.values[i,j]) for j in range(0, df.shape[1]) if str(df.values[i,j]) != 'nan'])
    
rec

rec1=[]
for lst in range(0, len(df)):
    rec1.append(list(filter(lambda x: str(x)!='nan', list(df.iloc[lst]))))
rec1


from efficient_apriori import apriori

itemsets, rules = apriori(rec, min_support=0.001, min_confidence=0.001)
type(itemsets)

itemsets.keys()
len(itemsets[5])



for r in rules:
    print (r)



#Method 2



from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


te = TransactionEncoder()
te_ary = te.fit(rec).transform(rec)
te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
df

df.to_csv('data,csv')


frequent_itemsets = apriori(df, min_support=0.001, use_colnames = True)
frequent_itemsets

# end time to calculation#%%%

pd.set_option('display.max_columns',None)


res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.000001)

print(res) #dataframe with confidence, lift, conviction and leverage metrics calculated

df_out = res[['antecedents', 'consequents', 'confidence', 'support', 'lift']]

df_out.to_csv("data1.csv")

df_out[df_out['confidence']>0.9]


# Case 2

import pandas as pd

df = pd.read_csv('online_store.csv')

df.dtypes
df.columns

df = df.drop(['Unnamed: 0','StockCode', 'InvoiceDate','UnitPrice', 'CustomerID', 'Country'], axis=1)

df.columns
df

df = df.dropna()
df

df = df[~df['InvoiceNo'].str.contains('C')]


basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket= df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')

basket.to_csv('basket.csv')


def encode_units(x):
    if x <= 0:
        return 0
    else:
        return 1

basket_sets = basket.applymap(encode_units)




from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


frequent_itemsets = apriori(basket_sets, min_support=0.02, use_colnames = True)
frequent_itemsets

# end time to calculation#%%%

pd.set_option('display.max_columns',None)


res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.000001)

print(res) #dataframe with confidence, lift, conviction and leverage metrics calculated

df_out = res[['antecedents', 'consequents', 'confidence', 'support', 'lift']]

df_out.to_csv("data1.csv")

df_out[df_out['confidence']>0.9]





























