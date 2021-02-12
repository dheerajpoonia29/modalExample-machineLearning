# not run

# #===== DATASET
# import pandas as pd
# df = pd.read_csv('Mall_Customers.csv')
# df.rename(columns={'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Spending_Score'}, inplace = True)
# print(df.head())

# #===== DATA EXPLORATION
# import seaborn as sns
# sns.pairplot(df[['Age','Income', 'Spending_Score']])


# #===== ML MODEL
# import sklearn.cluster as cluster
# model = cluster.KMeans(n_clusters=5 ,init="k-means++")
# model = model.fit(df[['Spending_Score','Income']])

# # Storing cluster prediction in csv
# df['Clusters'] = model.labels_
# df.to_csv('mallClusters.csv', index = False)  


# #===== ANALYSIS 
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.scatterplot(x="Spending_Score", y="Income",hue = 'Clusters',  data=df)


import numpy as np
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('store_data.csv')
store_data.head()

# In this updated output you will see that the first line is now treated as a record instead of header as shown below
store_data = pd.read_csv('store_data.csv', header=None)
print(store_data.head())

records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)



# print(len(association_rules))
print(association_rules)