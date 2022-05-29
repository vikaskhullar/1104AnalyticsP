#Topic ---- MB - store
#https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
#%%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from efficient_apriori import apriori


store_data = pd.read_csv('store_data1.csv', header=None)
store_data.head()
#%%%
'''
aa=store_data.to_numpy()
aa.shape
'''

store_data.shape


records = []

for i in range(0, 7501):
    print(i)
    records.append([str(store_data.values[i,j]) for j in range(0, 20) if str(store_data.values[i,j]) != 'nan'])

records



'''
# For Refernce
cnt = range(0,10)
a=[]
for i in cnt:
    if (i%2==0):
        a.append(i)
        
a

b = [i for i in cnt if(i%2==0)]
b
'''



itemsets, rules = apriori(records, min_support=0.01, min_confidence=0.01)

len(rules)

for r in rules:
    print (r)



rules1 = list(filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules))

rules1

for rule in rules1:
    print(rule)


#print(rules_rhs)

for rule in sorted(rules1, key=lambda rule: rule.support):
  print(rule) # Prints the rule and its confidence, support, lift, ...
  



association_results = list(rules)
association_results


 











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time
import logging



te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
te_ary.shape


df = pd.DataFrame(te_ary, columns=te.columns_)

df
df.to_csv("Rules.csv")


support_threshold=0.001


frequent_itemsets = apriori(df, min_support=0.01, use_colnames = True)
frequent_itemsets

confidence = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
print(confidence[1:3]) #dataframe with confidence, lift, conviction and leverage metrics calculated
print(confidence[['antecedents', 'consequents', 'support','confidence', 'lift']])

support = association_rules(frequent_itemsets, metric="support", min_threshold = 0)
print(support)
print(support[['antecedents', 'consequents', 'support','confidence']])

lift = association_rules(frequent_itemsets, metric="lift", min_threshold = 0)
a=lift[['antecedents', 'consequents', 'support','confidence','lift']]
a
print (a[1:10])



a[(a.confidence>0.5)]


