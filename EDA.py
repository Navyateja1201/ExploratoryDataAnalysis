#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_excel('data.xlsx')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df.nunique()


# In[11]:


data = df.copy()


# In[12]:


data = data.drop(columns = ['Unnamed: 0', 'CollegeID', 'CollegeCityID'])


# In[13]:


data['DOL'].value_counts()


# In[14]:


data['DOL'].replace('present','2015-12-31', inplace = True)


# In[15]:


data['DOL'] = pd.to_datetime(data['DOL'])
data['DOJ'] = pd.to_datetime(data['DOJ'])


# In[16]:


numerical_data = data.select_dtypes(['int64','float64'])


# In[17]:


categorical_data = data.select_dtypes(['object'])


# In[18]:


numerical_data.head()


# In[19]:


categorical_data.head()


# In[ ]:





# In[99]:


sns.boxplot(numerical_data['10percentage'])
print('10th Percentage')


# In[ ]:





# In[100]:


sns.boxplot(numerical_data['12percentage'])
print('12th Percentage')


# In[ ]:





# In[22]:


sns.boxplot(numerical_data['collegeGPA'])


# In[102]:


fig, ax = plt.subplots(3, figsize=(15,10))

sns.distplot(numerical_data['10percentage'], ax=ax[0])
sns.distplot(numerical_data['12percentage'], ax=ax[1])
sns.distplot(numerical_data['collegeGPA'], ax=ax[2])
import warnings
warnings.filterwarnings('ignore')


# In[24]:


plt.figure(figsize=(8,6))
sns.countplot(x = '12graduation', data = numerical_data)


# In[25]:


plt.figure(figsize = (5,5))

sns.countplot(x = 'CollegeTier', data = numerical_data)


# In[26]:


data['DOB'] = pd.to_datetime(data['DOB'])

data['Age'] = 2015 - data['DOB'].dt.year


# In[27]:


data['Age'].unique()


# In[28]:


data['Age'].describe()


# In[33]:


scores = ['English', 'Logical', 'Quant', 'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience', 'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg']
std_scores = ['conscientiousness', 'agreeableness', 'extraversion', 'nueroticism', 'openess_to_experience']


# In[34]:


for col in scores:
    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios = [2,1], figsize = (12,4))
    sns.boxplot(numerical_data[col], ax = ax2)
    sns.histplot(numerical_data[col], ax = ax1)


# In[35]:


for col in std_scores:
    fig, (ax1, ax2) = plt.subplots(1,2,width_ratios=[2,1], figsize=(12,4))
    sns.histplot(df[col],ax=ax1)
    sns.boxplot(df[col], ax=ax2)


# In[36]:


plt.figure(figsize=(5,5))
sns.countplot(data=df, x='12graduation')
plt.xticks(rotation=90)
plt.show()


# In[37]:


plt.figure(figsize=(5,5))
sns.countplot(data=df.dropna(subset='GraduationYear'), x='12graduation')
plt.xticks(rotation=90)
plt.show()


# In[40]:


plt.figure(figsize=(5,5))
sns.countplot(x=df['DOJ'].dt.year)
plt.xticks(rotation=90)
plt.show()


# In[50]:


df['Gender'].describe()


# In[52]:


df['Designation'].nunique()


# In[53]:


def refine_feature(input_val, input_list):
    if type(input_val) == str:
        for item in [i for i in input_list if len(i.split()) > 1]:
            if all([x in input_val for x in item.split()]):
                return item.title()
        for item in [i for i in input_list if len(i.split()) == 1]:
            if item in input_val:
                return item.title()
        if 'engineer' in input_val:
            return 'Hardware Engineer'
        try:
            matched_item = get_close_matches(input_val, input_list)[0]
            return matched_item.title()
        except:
            return 'Other'
    else:
        return np.nan


# In[55]:


df.columns


# In[57]:


data.groupby('Gender')['Age'].describe()


# In[59]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='Gender',y='Age', data=data, ax=ax1)
sns.boxplot(x='Gender',y='Age', data=data, ax=ax2)


# In[60]:


print('Marks obtained in 10th grade')
display(data.groupby('Gender')['10percentage'].describe())


# In[61]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='Gender',y='10percentage', data=data, ax=ax1)
sns.boxplot(x='Gender',y='10percentage', data=data, ax=ax2)


# In[62]:


print('Marks obtained in 12th grade')
display(data.groupby('Gender')['12percentage'].describe())


# In[63]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='Gender',y='12percentage', data=data, ax=ax1)
sns.boxplot(x='Gender',y='12percentage', data=data, ax=ax2)


# In[64]:


display(data.groupby('Gender')['collegeGPA'].describe())


# In[65]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='Gender',y='collegeGPA', data=data, ax=ax1)
sns.boxplot(x='Gender',y='collegeGPA', data=data, ax=ax2)


# In[67]:


crosstab = pd.crosstab(df["Gender"], df["Degree"])
crosstab


# In[68]:


plt.figure(figsize=(8, 6))
crosstab.plot(kind="bar", stacked=False, colormap="Set3")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Distribution of Degrees by Gender")
plt.xticks(rotation=0)
plt.tight_layout()


# In[71]:


df.groupby('Gender')['Salary'].describe()


# In[87]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='Gender',y='Salary', data=data, ax=ax1)
sns.boxplot(x='Gender',y='Salary', data=data, ax=ax2)


# In[73]:


df.groupby('CollegeTier')['collegeGPA'].describe()


# In[74]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='CollegeTier',y='collegeGPA', data=data, ax=ax1)
sns.boxplot(x='CollegeTier',y='collegeGPA', data=data, ax=ax2)


# In[75]:


df.groupby('CollegeTier')['Salary'].describe()


# In[76]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='CollegeTier',y='Salary', data=data, ax=ax1)
sns.boxplot(x='CollegeTier',y='Salary', data=data, ax=ax2)


# In[78]:


plt.figure(figsize=(8, 6))
plt.scatter(data["Age"], data["Salary"], color="blue", alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Scatter Plot: Age vs Salary")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[81]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data[["Age"]], data["Salary"])


# In[82]:


x_line = range(min(data["Age"]), max(data["Age"]) + 1)
y_line = model.predict(pd.DataFrame({"Age": x_line}))


# In[85]:


plt.scatter(data["Age"], data["Salary"], color="blue", alpha=0.7, label="Data Points")
plt.plot(x_line, y_line, color="red", label="Regression Line")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Scatter Plot with Regression Line: Age vs Salary")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[89]:


df.groupby('Degree')['Salary'].describe()


# In[90]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
sns.barplot(x='Degree',y='Salary', data=data, ax=ax1)
sns.boxplot(x='Degree',y='Salary', data=data, ax=ax2)


# In[91]:


from scipy.stats import chi2_contingency


# In[92]:


df_filtered = df[["Gender", "Specialization"]]


# In[93]:


contingency_table = pd.crosstab(df_filtered["Gender"], df_filtered["Specialization"])


# In[94]:


chi2, pval, dof, expcted = chi2_contingency(contingency_table)


# In[95]:


print("Chi-square statistic:", chi2)
print("p-value:", pval)
print("Degrees of freedom:", dof)


# In[96]:


if pval < 0.05:
    print("There is a statistically significant relationship between gender and specialization (p-value < 0.05).")
else:
    print("There is no sufficient evidence to claim a relationship between gender and specialization based on the data (p-value >= 0.05).")


# In[ ]:




