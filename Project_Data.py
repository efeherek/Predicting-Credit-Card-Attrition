import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Project/data/BankChurners.csv')


#deletin clientum id, naive bayes and avg utilization ratio columns as they are not relevant and/or raw data
df = df.iloc[:, 1:-2]

# map the target variable to binary values 
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# general look at the data
print(df.head())
print(df.info())

sns.set_style("whitegrid")

# How many customers are existing vs attiring?
plt.figure(figsize=(8, 5))
sns.countplot(x='Attrition_Flag', data=df, palette='viridis')
plt.title('Attrition Chart: (0: Existing, 1: Attired)')
plt.show()

# Age distribution and relevance to attrition
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Customer_Age', hue='Attrition_Flag', kde=True, element="step")
plt.title('Age distribtion by Attrition Stat')
plt.show()

# C. Analysis of Categorical Variables (Example: Gender)
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', hue='Attrition_Flag', data=df, palette='magma')
plt.title('Customer Attrition by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Status', labels=['Existing', 'Attrited'])
plt.show()

# D. Correlation Matrix (Relationship between numerical variables)
plt.figure(figsize=(12, 10))
# Selecting only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
# Calculating the correlation
corr_matrix = numeric_df.corr()

# Plotting the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()