# -*- coding: utf-8 -*-

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df = pd.read_csv('online_store.csv')
df.head()

df['Description'] = df['Description'].str.strip()
df
df['Description'].head(5)



df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

df.dtypes
df['InvoiceNo'] = df['InvoiceNo'].astype('str')

df = df[~df['InvoiceNo'].str.contains('C')]


basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket

basket.to_csv("in_basket.csv")


def encode_units(x):
    if x <= 0:
        return 0
    else:
        return 1

basket_sets = basket.applymap(encode_units)

basket_sets.drop('POSTAGE', inplace=True, axis=1)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)

frequent_itemsets

frequent_itemsets.to_csv("Freq_data.csv")


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.columns
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]


rules[rules['confidence']>0.7].plot(kind='bar')


r1= rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]


rules.to_csv("rules.csv")

basket['ALARM CLOCK BAKELIKE GREEN'].sum()

basket['ALARM CLOCK BAKELIKE RED'].sum()


basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets2 = basket2.applymap(encode_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

rules2[ (rules2['lift'] >= 4) &
        (rules2['confidence'] >= 0.5)]
