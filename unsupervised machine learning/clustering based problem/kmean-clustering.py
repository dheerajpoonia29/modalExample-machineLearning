#===== DATASET
import pandas as pd
df = pd.read_csv('Mall_Customers.csv')
df.rename(columns={'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Spending_Score'}, inplace = True)
print(df.head())

#===== DATA EXPLORATION
import seaborn as sns
sns.pairplot(df[['Age','Income', 'Spending_Score']])


#===== ML MODEL
import sklearn.cluster as cluster
model = cluster.KMeans(n_clusters=5 ,init="k-means++")
model = model.fit(df[['Spending_Score','Income']])

# Storing cluster prediction in csv
df['Clusters'] = model.labels_
df.to_csv('mallClusters.csv', index = False)  


#===== ANALYSIS 
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x="Spending_Score", y="Income",hue = 'Clusters',  data=df)
