#!/usr/bin/env python
# coding: utf-8

# In[112]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# ## Load the datasets

# In[115]:


train=pd.read_csv("C:\\Users\\dssm3\\Downloads\\train_ctrUa4K.csv")
train


# In[117]:


test=pd.read_csv("C:\\Users\\dssm3\\Downloads\\test_lAUu6dG.csv")
test


# ## Basic information

# ## Datatypes

# In[121]:


# DataTypes of train data 
train.dtypes


# In[123]:


# DataType of test data 
test.dtypes


# ### Summarize the data

# In[126]:


# for train data 
train.describe()


# In[128]:


# dor test data 
test.describe()


# ### Check duplicate rows or column

# In[131]:


# for train data 
dup_rows=train.duplicated()
dup_rows.sum()


# In[133]:


dup_col=train.columns.duplicated() 
dup_col.sum()


# In[135]:


# for test data 
dup_rows=test.duplicated()
dup_rows.sum()


# In[137]:


dup_col=test.columns.duplicated() 
dup_col.sum()


# ## Visualization of missing values 

# ## for test data 

# In[141]:


test.isna().sum()


# In[143]:


sns.heatmap(test.isna())
plt.show()


# ### For train data

# In[146]:


train.isna().sum()


# In[148]:


sns.heatmap(test.isna())
plt.show()


# ## Handle missing values

# #### For train data

# In[152]:


train.drop(['Loan_ID'],axis=1)


# In[154]:


train['Dependents']=pd.to_numeric(train['Dependents'],errors='coerce')


# In[ ]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[158]:


train.isna().sum()


# In[160]:


train.info()


# #### For test data 

# In[163]:


test['Dependents']=pd.to_numeric(test['Dependents'],errors='coerce')


# In[ ]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[167]:


test.isna().sum()


# In[169]:


test.info()


# ## Visualization of train dataset 

# In[172]:


train['Loan_Status'].value_counts()


# #### 422 peoples has got the approval for loan 

# In[175]:


train['Loan_Status'].value_counts().plot.bar()


# ### visualize each variable separately

# In[178]:


# Categorical variables
plt.figure(1) 
plt.subplot(2,2,1) 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
plt.subplot(2,2,2) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(2,2,3) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(2,2,4) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()


# #### * 80% of applicants in the dataset are male.
# #### * Around 65% of the applicants in the dataset are married.
# #### * About 15% of applicants in the dataset are self-employed.
# #### * About 85% of applicants have repaid their debts.

# In[181]:


# Ordinal variables
plt.figure(1) 
plt.subplot(1,3,1)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6),title='Dependents') 
plt.subplot(1,3,2)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
plt.subplot(1,3,3) 
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area') 
plt.show()


# #### * Most of the applicants don’t have dependents.
# #### * About 80% of the applicants are graduates.
# #### * Most of the applicants are from semi-urban areas.

# ### Distribution of Applicant income

# In[185]:


sns.histplot(train['ApplicantIncome'])
plt.show()


# #### Distribution of applicant income are towards the left which means it is not normally distributed. 

# ### Distribution of Loan amount 

# In[189]:


sns.histplot(train['LoanAmount'])
plt.show()


# #### Distribution of loan amount is fairly normal

# #### Following inferences can be made from the above two histplot 
# * Applicants with high incomes should have more chances of loan approval.
# * Loan approval should also depend on the loan amount. If the loan amount is less, the chances of loan approval should be high.

# ## model building 

# In[ ]:





# ### drop the Loan_ID variable

# In[196]:


train=train.drop('Loan_ID',axis=1) 
test_dup=test.drop('Loan_ID',axis=1)


# In[198]:


X = train.drop('Loan_Status',axis=1) 
y = train.Loan_Status


# In[200]:


X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test_dup=pd.get_dummies(test_dup)


# ### use the train_test_split function 

# In[203]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)


# In[205]:


#import LogisticRegression and accuracy_score from sklearn and fit the logistic regression model. 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score


# In[ ]:


model = LogisticRegression() 
model.fit(x_train, y_train)


# In[209]:


# predict the Loan_Status for the validation set and calculate its accuracy. 
pred_cv = model.predict(x_cv)


# In[211]:


#calculate how accurate our predictions are by calculating the accuracy. 
accuracy_score(y_cv,pred_cv)


# ###  So our predictions are almost 80% accurate, i.e. we have identified 80% of the loan status correctly. 
# 

# In[214]:


# make predictions for the test dataset. 
pred_test = model.predict(test_dup)


# #### import the submission file which we have to submit on the solution checker.

# In[217]:


submission=pd.read_csv("C:\\Users\\dssm3\\Downloads\\sample_submission_49d68Cx.csv")


# #### We only need the Loan_ID and the corresponding Loan_Status for the final submission. we will fill these columns with the Loan_ID of the test dataset and the predictions that we made, i.e., pred_test respectively.

# In[220]:


submission['Loan_Status']=pred_test 
submission['Loan_ID']=test['Loan_ID']


# In[ ]:


# we need predictions in Y and N. So let’s convert 1 and 0 to Y and N. 
submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[228]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[230]:


import os
print(os.getcwd())


# In[ ]:




