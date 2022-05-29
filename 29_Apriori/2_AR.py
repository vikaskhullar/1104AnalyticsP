#Topic ---- Association Rule Analysis 
!pip install mlxtend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


transactions = [['milk', 'water'], ['milk', 'bread'], ['milk','bread','water']]
transactions
#%%%

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
df

# apriori

frequent_itemsets = apriori(df, min_support=0.0000001, use_colnames = True)
frequent_itemsets

# end time to calculation#%%%

pd.set_option('display.max_columns',None)


res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.000001)

print(res) #dataframe with confidence, lift, conviction and leverage metrics calculated



df_out = res[['antecedents', 'consequents', 'confidence', 'support', 'lift']]

df_out

df_out[(df_out['confidence']==1.0) & (df_out['support']>0.6)]






