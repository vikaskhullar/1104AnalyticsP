#Topic ---- MB - store
#https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
#%%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



store_data = pd.read_csv('store_data1.csv')
store_data.shape

records = []

for i in range(0, 7501):
    print(i)
    records.append([str(store_data.values[i,j]) for j in range(0, 20) if str(store_data.values[i,j]) != 'nan'])

records


te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
te_ary.shape


df = pd.DataFrame(te_ary, columns=te.columns_)

df.to_csv('convert.csv')

support_threshold=0.001

frequent_itemsets = apriori(df, min_support=support_threshold, use_colnames = True)

confidence = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
print(confidence) #dataframe with confidence, lift, conviction and leverage metrics calculated
df1 = confidence[['antecedents', 'consequents', 'support','confidence']]

df1.to_csv('association.csv')



df1[df1['confidence']>0.25]



support = association_rules(frequent_itemsets, metric="support", min_threshold = 0)
print(support)
print(support[['antecedents', 'consequents', 'support','confidence']])

lift = association_rules(frequent_itemsets, metric="lift", min_threshold = 0)
a=lift[['antecedents', 'consequents', 'support','confidence','lift']]

print (a[1:10])

a[(a.lift>12) & (a.confidence>0.5)][1:2]



