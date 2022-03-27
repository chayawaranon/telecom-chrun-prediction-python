import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chisquare, chi2_contingency

df = pd.read_csv("C:/Users/aimza/Desktop/Project Chrun/telecom_users.csv")
# print(df.info())

df_dev = df.iloc[:, 2:].copy()

le = LabelEncoder()
# Partner, Dependents, PhoneService, PaperlessBilling, Churn have Yes (1) , No (0) output
le.fit(['Yes', 'No'])
yes_no_classes = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for service in yes_no_classes:
    df_dev[service] = le.transform(df_dev[service]).astype('object')

# change gender Male = M, Female = F
df_dev['gender'] = ['M' if x == 'Male' else 'F' for x in df_dev['gender']]

# OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies has Yes (2), No (0),
# No Internet (1)
le_online = LabelEncoder()
onlineServices = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

le_online.fit(df_dev['OnlineSecurity'])
for service in onlineServices:
    df_dev[service] = le_online.transform(df_dev[service]).astype('object')

# fill 0 in TotalCharges for tenure = 0 rows
df_dev.loc[df_dev['tenure'] == 0, 'TotalCharges'] = 0

# set up appropriate type for each attributes
df_dev['SeniorCitizen'] = df_dev['SeniorCitizen'].astype('object')
df_dev['TotalCharges'] = df_dev['TotalCharges'].astype('float64')

# df_dev_numeric = df_dev.loc[:, ['tenure', 'MonthlyCharges', 'TotalCharges','Churn']].copy()

# find pearson's correlation
corr = df_dev.corr()
# print(corr.columns, corr.index)

# visualization
# sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
# plt.show()
#
# for feature in df_dev.columns:
#     if feature not in ['tenure', 'MonthlyCharges', 'TotalCharges']:
#         sns.countplot(x=df_dev[feature], data=df_dev)
#         plt.show()

# chi square test
# number of churn and not churn
n_yes = len(df_dev.loc[df_dev['Churn'] == 1])
n_no = len(df_dev.loc[df_dev['Churn'] == 0])

df_chiSquare = pd.DataFrame({
    'column': [],
    'chiSquare': [],
    'p-value': [],
    'relation': [],
})
alpha = 0.05

for column in df_dev.columns:
    obs_values = []
    if column not in ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']:
        # print(column)
        ls = df_dev.groupby([column, 'Churn']).count().reset_index()
        for unique in df_dev[column].unique():
            yes = ls.loc[(ls[column] == unique) & (ls['Churn'] == 1)]['TotalCharges'].values.tolist()
            no = ls.loc[(ls[column] == unique) & (ls['Churn'] == 0)]['TotalCharges'].values.tolist()
            obs_values.append(no + yes)
        stat, p, dof, expected = chi2_contingency(obs_values)
        if p <= alpha:
            df_chiSquare = df_chiSquare.append({
                'column': column,
                'chiSquare': stat,
                'p-value': p,
                'relation': 'Yes'
            }, ignore_index=True)
        else:
            df_chiSquare = df_chiSquare.append({
                'column': column,
                'chiSquare': stat,
                'p-value': p,
                'relation': 'No'
            }, ignore_index=True)
print('\nChi square summaries dataframe\n')
print(df_chiSquare)

# kulczynski measure
df_kulc = pd.DataFrame({
    'column': [],
    'class': [],
    'kulc': [],
    'IR': [],
    'balance': [],
    'relation': [],
})

for column in df_dev.columns:
    if column not in ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']:
        kulc_ls = df_dev.groupby([column, 'Churn']).count().reset_index()
        for unique in df_dev[column].unique():
            size_unique = len(df_dev.loc[df_dev[column] == unique])
            sup_unique_yes = kulc_ls.loc[(kulc_ls[column] == unique) & (kulc_ls['Churn'] == 1)]['TotalCharges'].values[
                0]
            kulc = (sup_unique_yes / 2) * ((1 / size_unique) + (1 / n_yes))
            ir = abs(n_yes - size_unique) / (n_yes + size_unique - sup_unique_yes)

            kulc_status = str()
            ir_status = str()
            if abs(kulc - 0.5) < 0.1:
                kulc_status = 'no relation with yes in churn class'
            elif kulc < 0.5:
                kulc_status = 'negative relation with yes in churn class'
            else:
                kulc_status = 'positive relation with yes in churn class'

            if ir >= 0.8:
                ir_status = 'high imbalance with yes in churn class'
            elif ir >= 0.5:
                ir_status = 'imbalance with yes in churn class'
            elif 0.5 > ir >= 0.3:
                ir_status = 'low imbalance with yes in churn class'
            else:
                ir_status = 'very low imbalance or balance with yes in churn class'

            df_kulc = df_kulc.append({
                'column': column,
                'class': unique,
                'kulc': kulc,
                'IR': ir,
                'balance': ir_status,
                'relation': kulc_status,
            },ignore_index=True)

# show all columns
pd.set_option('display.max_columns', None)
print('\nKULC and IR dataframe\n')
print(df_kulc.head())

# summaries of kulc and ir
df_kulc_sum = df_kulc.groupby(['column']).agg(['mean']).reset_index()
print('\nKULC and IR summaries dataframe\n')
print(df_kulc_sum)