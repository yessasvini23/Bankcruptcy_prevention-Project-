#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr,kendalltau
from matplotlib import rc_params
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings("ignore")


# In[3]:


bank=pd.read_csv("bankruptcy-prevention.csv",delimiter=";")


# In[4]:


bank


# In[5]:


bank.info()


# In[6]:


bank.shape


# In[7]:


bank.size


# In[8]:


bank.isnull().sum()


# In[9]:


bank.describe(include="all")


# In[10]:


bank[bank.duplicated()].shape


# In[11]:


bank[bank.duplicated()]


# In[12]:


bank1=bank.drop_duplicates()


# In[13]:


bank1


# In[14]:


bank1[" class"].unique()


# In[15]:


bank1[" class"].value_counts()


# In[16]:


# Remaining columns to avoid the error

bank1=bank1.rename(columns={bank.columns[6]: "class_value"})
bank1=bank1.rename(columns={bank.columns[5]: "operating_risk"})
bank1=bank1.rename(columns={bank.columns[4]: "competitiveness"})
bank1=bank1.rename(columns={bank.columns[3]: "credibility"})
bank1=bank1.rename(columns={bank.columns[2]: "financial_flexibility"})
bank1=bank1.rename(columns={bank.columns[1]: "management_risk"})


# In[17]:


bank1.head()


# In[18]:


bank1.tail()


# Here are we taking a bankruptcy = 0 and Non-bankruptcy=1 by encoding data

# In[19]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
bank1["class_value"]=label_encoder.fit_transform(bank1["class_value"])


# In[20]:


from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
bank[" class"]=encode.fit_transform(bank[" class"])


# In[21]:


bank


# In[22]:


corr_values=bank1.corr()


# In[23]:


corr_values


# In[24]:


sns.heatmap(corr_values,annot=True,square=True,cmap="viridis")
plt.yticks(rotation=0)
plt.show()


# In[25]:


features=bank1.columns[0:6]


# In[26]:


features


# Spearmans Correlation

# In[27]:


for feature in features:
  coef,p=spearmanr(bank1[feature],bank1.class_value)
  print("Correlation of %s" %feature,coef)


# Kendall's Correlation

# In[28]:


for feature in features:
  coef,p=kendalltau(bank1[feature],bank1.class_value)
  print("Correlation of %s" %feature,coef)


# In[29]:


bank1.columns
pd.crosstab(bank1.industrial_risk,bank1.class_value)


# In[30]:


pd.crosstab(bank1.industrial_risk,bank1.class_value,normalize="index")


# In[31]:


pd.crosstab(bank1.management_risk,bank1.class_value)


# In[32]:


pd.crosstab(bank1.management_risk,bank1.class_value,normalize="index")


# In[33]:


pd.crosstab(bank1.financial_flexibility,bank1.class_value)


# In[34]:


pd.crosstab(bank1.financial_flexibility,bank1.class_value,normalize="index")


# In[35]:


pd.crosstab(bank1.credibility,bank1.class_value)


# In[36]:


pd.crosstab(bank1.credibility,bank1.class_value,normalize="index")


# In[37]:


pd.crosstab(bank1.competitiveness,bank1.class_value)


# In[38]:


pd.crosstab(bank1.competitiveness,bank1.class_value,normalize="index")


# In[39]:


pd.crosstab(bank1.operating_risk,bank1.class_value)


# In[40]:


pd.crosstab(bank1.operating_risk,bank1.class_value,normalize="index")


# In[41]:


pd.crosstab(bank1.financial_flexibility, bank1.competitiveness, normalize='index')


# In[42]:


#Pairplot to understand relationships between variables within a dataset
sns.pairplot(bank, hue = ' class')


# In[43]:


a =bank[' class'].value_counts()[0]
b =bank[' class'].value_counts()[1]


fig1, ax1 = plt.subplots(figsize=(8, 6))
label = ['bankruptcy', 'non-bankruptcy']
count = [a, b]
colors = ['red', 'yellowgreen']
explode = (0, 0.1)  # explode 2nd slice
plt.pie(count, labels=label, autopct='%0.2f%%', explode=explode, colors=colors,shadow=True, startangle=90)
plt.show()


# In[44]:


bank.plot(kind='box', subplots=True, layout=(4,4), figsize=(12,10))
plt.show()


# In[45]:


sns.set_theme();
ax = sns.distplot(bank)


# In[46]:


#Densityplot to observe the distribution of a variable in a dataset
bank.plot(kind='density', subplots=True, layout=(2,4), sharex=False, figsize=(20,15))
plt.show()


# In[47]:


bank1.groupby('class_value')[features].mean().T.plot(figsize=(12,8))


# In[48]:


for feature in features:
    sns.violinplot(x='class_value', y=feature, data=bank1, inner=None, color='lightgray')
    sns.stripplot(x='class_value', y=feature, data=bank1, size=2, jitter=True)
    plt.ylabel(feature)
    plt.title("%s vs Bankruptcy" % feature)
    plt.show()


# In[49]:


for feature in features:
    fig, ax = plt.subplots()

    ax.hist(bank1[bank1["class_value"]==1][feature], bins=15, alpha=0.5, color="blue", label="Not Bankrupted")
    ax.hist(bank1[bank1["class_value"]==0][feature], bins=15, alpha=0.5, color="green", label="Bankrupted")

    ax.set_xlabel(feature)
    ax.set_ylabel("Business count")
    ax.set_title("%s vs Bankruptcy"%feature)

    ax.legend();


# In[50]:


# Density Estimate Plots
for feature in features:
    fig, ax = plt.subplots()

    sns.kdeplot(bank1[bank1["class_value"]==1][feature], shade=True, color="blue", label="Not Bankrupted", ax=ax)
    sns.kdeplot(bank1[bank1["class_value"]==0][feature], shade=True, color="green", label="Bankrupted", ax=ax)

    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.set_title("%s vs Bankruptcy"%feature)
    #fig.suptitle("Financial Flexibility vs. Bankruptcy");
    ax.legend();


# In[51]:


# Stacked Bar Charts for checking proportion
for feature in features:
    counts_df = bank1.groupby([feature, "class_value"])["competitiveness"].count().unstack()
    Bankruptcy_df = counts_df.T.div(counts_df.T.sum()).T
    g=Bankruptcy_df.plot(kind="bar", stacked=True, color=["green", "red"]).set(title="%s vs Bankruptcy"%feature)

    plt.gcf().set_size_inches(6,6)
    plt.legend(title='Bankruptcy', loc='upper right', labels=['Bankrupted', 'Not Bankrupted'])
    plt.xlabel(feature)
    plt.ylabel("Proportion")


# Model Building

# In[52]:


bank1.head()


# In[53]:


X=bank1.iloc[:,0:6]


# In[54]:


X


# In[55]:


y=bank1["class_value"]


# In[56]:


y


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, classification_report, roc_auc_score, roc_curve



# In[60]:


# Logestic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)


# In[61]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

log_train_acc = accuracy_score(y_train, y_pred_train)
log_test_acc = accuracy_score(y_test, y_pred_test)

log_train_f1 = f1_score(y_train, y_pred_train)
log_test_f1 = f1_score(y_test, y_pred_test)

log_train_prec = precision_score(y_train, y_pred_train)
log_test_prec = precision_score(y_test, y_pred_test)

log_train_roc = roc_auc_score(y_train, y_pred_train)
log_test_roc = roc_auc_score(y_test, y_pred_test)

log_train_cm = confusion_matrix(y_train, y_pred_train)
log_test_cm = confusion_matrix(y_test, y_pred_test)


# In[62]:


#Printing all the meeasuring metric results
print('Logistic Regression training accuracy is', log_train_acc)
print('Logistic Regression testing accuracy is', log_test_acc)
print('--------------------------------------------------------')
print('Logistic Regression training F1 Score is', log_train_f1)
print('Logistic Regression testing F1 Score is', log_test_f1)
print('--------------------------------------------------------')
print('Logistic Regression training Precision is', log_train_prec)
print('Logistic Regression testing Precision is', log_test_prec)
print('--------------------------------------------------------')
print('Logistic Regression training ROC-AUC Score is', log_train_roc)
print('Logistic Regression testing ROC-AUC Score is', log_test_roc)
print('--------------------------------------------------------')
print('Logistic Regression training Confusion Matrix is')
print(log_train_cm)
print('--------------------------------------------------------')
print('Logistic Regression testing Confusion Matrix is')
print(log_test_cm)


# In[63]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {log_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[64]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[65]:


#Decision Tree Model
dec = DecisionTreeClassifier(max_depth=5)
monotonic_feature = 0
dec.fit(X_train, y_train)


# In[66]:


# Training and Testing the model and Measuring Accuracies
y_pred_train_dec = dec.predict(X_train)
y_pred_test_dec = dec.predict(X_test)

dec_train_acc = accuracy_score(y_train, y_pred_train_dec)
dec_test_acc = accuracy_score(y_test, y_pred_test_dec)

dec_train_f1 = f1_score(y_train, y_pred_train_dec)
dec_test_f1 = f1_score(y_test, y_pred_test_dec)

dec_train_prec = precision_score(y_train, y_pred_train_dec)
dec_test_prec = precision_score(y_test, y_pred_test_dec)

dec_train_roc = roc_auc_score(y_train, y_pred_train_dec)
dec_test_roc = roc_auc_score(y_test, y_pred_test_dec)

dec_train_cm = confusion_matrix(y_train, y_pred_train_dec)
dec_test_cm = confusion_matrix(y_test, y_pred_test_dec)


# In[67]:


#Printing all the meeasuring metric results
print('Decision tree training accuracy is', dec_train_acc)
print('Decision tree testing accuracy is', dec_test_acc)
print(" ")
print('Decision tree training F1 Score is', dec_train_f1)
print('Decision tree testing F1 Score is', dec_test_f1)
print(" ")
print('Decision tree training Precision is', dec_train_prec)
print('Decision tree testing Precision is', dec_test_prec)
print(" ")
print('Decision tree training ROC-AUC Score is', dec_train_roc)
print('Decision tree testing ROC-AUC Score is', dec_test_roc)
print(" ")
print('Decision tree training Confusion Matrix is')
print(dec_train_cm)
print(" ")
print('Decision tree testing Confusion Matrix is')
print(dec_test_cm)


# In[68]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {dec_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[69]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[70]:


#KNN Classifier Model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# In[71]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

knn_train_acc = accuracy_score(y_train, y_pred_train)
knn_test_acc = accuracy_score(y_test, y_pred_test)

knn_train_f1 = f1_score(y_train, y_pred_train)
knn_test_f1 = f1_score(y_test, y_pred_test)

knn_train_prec = precision_score(y_train, y_pred_train)
knn_test_prec = precision_score(y_test, y_pred_test)

knn_train_roc = roc_auc_score(y_train, y_pred_train)
knn_test_roc = roc_auc_score(y_test, y_pred_test)

knn_train_cm = confusion_matrix(y_train, y_pred_train)
knn_test_cm = confusion_matrix(y_test, y_pred_test)


# In[72]:


#Printing all the meeasuring metric results
print('KNN training accuracy is', knn_train_acc)
print('KNN testing accuracy is', knn_test_acc)
print(" ")
print('KNN training F1 Score is', knn_train_f1)
print('KNN testing F1 Score is', knn_test_f1)
print(" ")
print('KNN training Precision is', knn_train_prec)
print('KNN testing Precision is', knn_test_prec)
print(" ")
print('KNN training ROC-AUC Score is', knn_train_roc)
print('KNN testing ROC-AUC Score is', knn_test_roc)
print(" ")
print('KNN training Confusion Matrix is')
print(knn_train_cm)
print(" ")
print('KNN testing Confusion Matrix is')
print(knn_test_cm)


# In[73]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {knn_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[74]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[75]:


#Random Forest Classifier Model
rnd = RandomForestClassifier(max_depth=5)
rnd.fit(X_train, y_train)


# In[76]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = rnd.predict(X_train)
y_pred_test = rnd.predict(X_test)

rnd_train_acc = accuracy_score(y_train, y_pred_train)
rnd_test_acc = accuracy_score(y_test, y_pred_test)

rnd_train_f1 = f1_score(y_train, y_pred_train)
rnd_test_f1 = f1_score(y_test, y_pred_test)

rnd_train_prec = precision_score(y_train, y_pred_train)
rnd_test_prec = precision_score(y_test, y_pred_test)

rnd_train_roc = roc_auc_score(y_train, y_pred_train)
rnd_test_roc = roc_auc_score(y_test, y_pred_test)

rnd_train_cm = confusion_matrix(y_train, y_pred_train)
rnd_test_cm = confusion_matrix(y_test, y_pred_test)


# In[77]:


#Printing all the meeasuring metric results
print('Random Forest training accuracy is', rnd_train_acc)
print('Random Forest testing accuracy is', rnd_test_acc)
print(" ")
print('Random Forest training F1 Score is', rnd_train_f1)
print('Random Forest testing F1 Score is', rnd_test_f1)
print(" ")
print('Random Forest training Precision is', rnd_train_prec)
print('Random Forest testing Precision is', rnd_test_prec)
print(" ")
print('Random Forest training ROC-AUC Score is', rnd_train_roc)
print('Random Forest testing ROC-AUC Score is', rnd_test_roc)
print(" ")
print('Random Forest training Confusion Matrix is')
print(rnd_train_cm)
print(" ")
print('Random Forest testing Confusion Matrix is')
print(rnd_test_cm)


# In[78]:


#Ploting decision tree using Random forest model
fig = plt.figure(figsize=(12,10))
_ = tree.plot_tree(rnd.estimators_[9], feature_names= list(X),filled=True, max_depth=5)


# In[79]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {rnd_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[80]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[81]:


#Bagging Classifier
bag = BaggingClassifier(estimator=RandomForestClassifier(max_depth = 5), n_estimators=10)
bag.fit(X_train, y_train)


# In[82]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = bag.predict(X_train)
y_pred_test = bag.predict(X_test)

bag_train_acc = accuracy_score(y_train, y_pred_train)
bag_test_acc = accuracy_score(y_test, y_pred_test)

bag_train_f1 = f1_score(y_train, y_pred_train)
bag_test_f1 = f1_score(y_test, y_pred_test)

bag_train_prec = precision_score(y_train, y_pred_train)
bag_test_prec = precision_score(y_test, y_pred_test)

bag_train_roc = roc_auc_score(y_train, y_pred_train)
bag_test_roc = roc_auc_score(y_test, y_pred_test)

bag_train_cm = confusion_matrix(y_train, y_pred_train)
bag_test_cm = confusion_matrix(y_test, y_pred_test)


# In[83]:


#Printing all the meeasuring metric results
print('Bagging training accuracy is', bag_train_acc)
print('Bagging testing accuracy is', bag_test_acc)
print(" ")
print('Bagging training F1 Score is', bag_train_f1)
print('Bagging testing F1 Score is', bag_test_f1)
print(" ")
print('Bagging training Precision is', bag_train_prec)
print('Bagging testing Precision is', bag_test_prec)
print(" ")
print('Bagging training ROC-AUC Score is', bag_train_roc)
print('Bagging testing ROC-AUC Score is', bag_test_roc)
print(" ")
print('Bagging training Confusion Matrix is')
print(bag_train_cm)
print(" ")
print('Bagging testing Confusion Matrix is')
print(bag_test_cm)


# In[84]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {bag_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[85]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[86]:


#Ada Boost
ada = AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=5))
ada.fit(X_train, y_train)


# In[87]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = ada.predict(X_train)
y_pred_test = ada.predict(X_test)

ada_train_acc = accuracy_score(y_train, y_pred_train)
ada_test_acc = accuracy_score(y_test, y_pred_test)

ada_train_f1 = f1_score(y_train, y_pred_train)
ada_test_f1 = f1_score(y_test, y_pred_test)

ada_train_prec = precision_score(y_train, y_pred_train)
ada_test_prec = precision_score(y_test, y_pred_test)

ada_train_roc = roc_auc_score(y_train, y_pred_train)
ada_test_roc = roc_auc_score(y_test, y_pred_test)

ada_train_cm = confusion_matrix(y_train, y_pred_train)
ada_test_cm = confusion_matrix(y_test, y_pred_test)


# In[88]:


#Printing all the meeasuring metric results
print('AdaBoost training accuracy is', ada_train_acc)
print('AdaBoost testing accuracy is', ada_test_acc)
print(" ")
print('AdaBoost training F1 Score is', ada_train_f1)
print('AdaBoost testing F1 Score is', ada_test_f1)
print(" ")
print('AdaBoost training Precision is', ada_train_prec)
print('AdaBoost testing Precision is', ada_test_prec)
print(" ")
print('AdaBoost training ROC-AUC Score is', ada_train_roc)
print('AdaBoost testing ROC-AUC Score is', ada_test_roc)
print(" ")
print('AdaBoost training Confusion Matrix is')
print(ada_train_cm)
print(" ")
print('AdaBoost testing Confusion Matrix is')
print(ada_test_cm)


# In[89]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {ada_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[90]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[91]:


#Gradient Boosting
grad = GradientBoostingClassifier(learning_rate=0.01)
grad.fit(X_train, y_train)


# In[92]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = grad.predict(X_train)
y_pred_test = grad.predict(X_test)

grad_train_acc = accuracy_score(y_train, y_pred_train)
grad_test_acc = accuracy_score(y_test, y_pred_test)

grad_train_f1 = f1_score(y_train, y_pred_train)
grad_test_f1 = f1_score(y_test, y_pred_test)

grad_train_prec = precision_score(y_train, y_pred_train)
grad_test_prec = precision_score(y_test, y_pred_test)

grad_train_roc = roc_auc_score(y_train, y_pred_train)
grad_test_roc = roc_auc_score(y_test, y_pred_test)

grad_train_cm = confusion_matrix(y_train, y_pred_train)
grad_test_cm = confusion_matrix(y_test, y_pred_test)


# In[93]:


#Printing all the meeasuring metric results
print('Gradient Boost training accuracy is', grad_train_acc)
print('Gradient Boost testing accuracy is', grad_test_acc)
print(" ")
print('Gradient Boost training F1 Score is', grad_train_f1)
print('Gradient Boost testing F1 Score is', grad_test_f1)
print(" ")
print('Gradient Boost training Precision is', grad_train_prec)
print('Gradient Boost testing Precision is', grad_test_prec)
print(" ")
print('Gradient Boost training ROC-AUC Score is', grad_train_roc)
print('Gradient Boost testing ROC-AUC Score is', grad_test_roc)
print(" ")
print('Gradient Boost training Confusion Matrix is')
print(grad_train_cm)
print(" ")
print('Gradient Boost testing Confusion Matrix is')
print(grad_test_cm)


# In[94]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {grad_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[95]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[96]:


svcl = SVC(kernel= "linear")
svcl.fit(X_train , y_train)


# In[97]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = svcl.predict(X_train)
y_pred_test = svcl.predict(X_test)

svcl_train_acc = accuracy_score(y_train, y_pred_train)
svcl_test_acc = accuracy_score(y_test, y_pred_test)

svcl_train_f1 = f1_score(y_train, y_pred_train)
svcl_test_f1 = f1_score(y_test, y_pred_test)

svcl_train_prec = precision_score(y_train, y_pred_train)
svcl_test_prec = precision_score(y_test, y_pred_test)

svcl_train_roc = roc_auc_score(y_train, y_pred_train)
svcl_test_roc = roc_auc_score(y_test, y_pred_test)

svcl_train_cm = confusion_matrix(y_train, y_pred_train)
svcl_test_cm = confusion_matrix(y_test, y_pred_test)


# In[98]:



#Printing all the meeasuring metric results
print('SVC Linear Kernel training accuracy is', svcl_train_acc)
print('SVC Linear Kernel testing accuracy is', svcl_test_acc)
print(" ")
print('SVC Linear Kernel training F1 Score is', svcl_train_f1)
print('SVC Linear Kernel testing F1 Score is', svcl_test_f1)
print(" ")
print('SVC Linear Kernel training Precision is', svcl_train_prec)
print('SVC Linear Kernel testing Precision is', svcl_test_prec)
print(" ")
print('SVC Linear Kernel training ROC-AUC Score is', svcl_train_roc)
print('SVC Linear Kernel testing ROC-AUC Score is', svcl_test_roc)
print(" ")
print('SVC Linear Kernel training Confusion Matrix is')
print(svcl_train_cm)
print(" ")
print('SVC Linear Kernel testing Confusion Matrix is')
print(svcl_test_cm)


# In[99]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {svcl_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[100]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[101]:


svcp = SVC(kernel= "poly")
svcp.fit(X_train , y_train)


# In[102]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = svcp.predict(X_train)
y_pred_test = svcp.predict(X_test)

svcp_train_acc = accuracy_score(y_train, y_pred_train)
svcp_test_acc = accuracy_score(y_test, y_pred_test)

svcp_train_f1 = f1_score(y_train, y_pred_train)
svcp_test_f1 = f1_score(y_test, y_pred_test)

svcp_train_prec = precision_score(y_train, y_pred_train)
svcp_test_prec = precision_score(y_test, y_pred_test)

svcp_train_roc = roc_auc_score(y_train, y_pred_train)
svcp_test_roc = roc_auc_score(y_test, y_pred_test)

svcp_train_cm = confusion_matrix(y_train, y_pred_train)
svcp_test_cm = confusion_matrix(y_test, y_pred_test)


# In[103]:


#Printing all the meeasuring metric results
print('SVC Polynomial Kernel training accuracy is', svcp_train_acc)
print('SVC Polynomial Kernel testing accuracy is', svcp_test_acc)
print(" ")
print('SVC Polynomial Kernel training F1 Score is', svcp_train_f1)
print('SVC Polynomial Kernel testing F1 Score is', svcp_test_f1)
print(" ")
print('SVC Polynomial Kernel training Precision is', svcp_train_prec)
print('SVC Polynomial Kernel testing Precision is', svcp_test_prec)
print(" ")
print('SVC Polynomial Kernel training ROC-AUC Score is', svcp_train_roc)
print('SVC Polynomial Kernel testing ROC-AUC Score is', svcp_test_roc)
print(" ")
print('SVC Polynomial Kernel training Confusion Matrix is')
print(svcp_train_cm)
print(" ")
print('SVC Polynomial Kernel testing Confusion Matrix is')
print(svcp_test_cm)


# In[104]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {svcp_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[105]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[106]:


svcr = SVC(C= 20, gamma = 50)
svcr.fit(X_train , y_train)


# In[107]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = svcr.predict(X_train)
y_pred_test = svcr.predict(X_test)

svcr_train_acc = accuracy_score(y_train, y_pred_train)
svcr_test_acc = accuracy_score(y_test, y_pred_test)

svcr_train_f1 = f1_score(y_train, y_pred_train)
svcr_test_f1 = f1_score(y_test, y_pred_test)

svcr_train_prec = precision_score(y_train, y_pred_train)
svcr_test_prec = precision_score(y_test, y_pred_test)

svcr_train_roc = roc_auc_score(y_train, y_pred_train)
svcr_test_roc = roc_auc_score(y_test, y_pred_test)

svcr_train_cm = confusion_matrix(y_train, y_pred_train)
svcr_test_cm = confusion_matrix(y_test, y_pred_test)


# In[108]:


#Printing all the meeasuring metric results
print('SVC RBF Kernel training accuracy is', svcr_train_acc)
print('SVC RBF Kernel testing accuracy is', svcr_test_acc)
print(" ")
print('SVC RBF Kernel training F1 Score is', svcr_train_f1)
print('SVC RBF Kernel testing F1 Score is', svcr_test_f1)
print(" ")
print('SVC RBF Kernel training Precision is', svcr_train_prec)
print('SVC RBF Kernel testing Precision is', svcr_test_prec)
print(" ")
print('SVC RBF Kernel training ROC-AUC Score is', svcr_train_roc)
print('SVC RBF Kernel testing ROC-AUC Score is', svcr_test_roc)
print(" ")
print('SVC RBF Kernel training Confusion Matrix is')
print(svcr_train_cm)
print(" ")
print('SVC RBF Kernel testing Confusion Matrix is')
print(svcr_test_cm)


# In[109]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {svcr_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[110]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[111]:


estimators = [('log', LogisticRegression(max_iter=500)), ('rnd', RandomForestClassifier(max_depth=5)), ('svcr', SVC(C= 20, gamma = 50))]
stack = StackingClassifier(estimators=estimators)
stack.fit(X_train, y_train)


# In[112]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = stack.predict(X_train)
y_pred_test = stack.predict(X_test)

stack_train_acc = accuracy_score(y_train, y_pred_train)
stack_test_acc = accuracy_score(y_test, y_pred_test)

stack_train_f1 = f1_score(y_train, y_pred_train)
stack_test_f1 = f1_score(y_test, y_pred_test)

stack_train_prec = precision_score(y_train, y_pred_train)
stack_test_prec = precision_score(y_test, y_pred_test)

stack_train_roc = roc_auc_score(y_train, y_pred_train)
stack_test_roc = roc_auc_score(y_test, y_pred_test)

stack_train_cm = confusion_matrix(y_train, y_pred_train)
stack_test_cm = confusion_matrix(y_test, y_pred_test)


# In[113]:


#Printing all the meeasuring metric results
print('Stacking training accuracy is', stack_train_acc)
print('Stacking testing accuracy is', stack_test_acc)
print()
print('Stacking training F1 Score is', stack_train_f1)
print('Stacking testing F1 Score is', stack_test_f1)
print()
print('Stacking training Precision is', stack_train_prec)
print('Stacking testing Precision is', stack_test_prec)
print()
print('Stacking training ROC-AUC Score is', stack_train_roc)
print('Stacking testing ROC-AUC Score is', stack_test_roc)
print()
print('Stacking training Confusion Matrix is')
print(stack_train_cm)
print()
print('Stacking testing Confusion Matrix is')
print(stack_test_cm)


# In[114]:


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

# Ploting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'AUC = {stack_test_roc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[115]:


results = {
    'Model Name': ['Logistic Regression','Decision Tree', 'KNN', 'Random Forest', 'Bagging', 'AdaBoost', 'Gradient Boost', 'SVC Linear Kernel', 'SVC Polynomial Kernel', 'SVC RBF Kernel', 'Stacking'],
    'Training Accuracy': [log_train_acc, dec_train_acc, knn_train_acc, rnd_train_acc, bag_train_acc, ada_train_acc, grad_train_acc, svcl_train_acc, svcp_train_acc, svcr_train_acc, stack_train_acc],
    'Testing Accuracy': [log_test_acc, dec_test_acc, knn_test_acc, rnd_test_acc, bag_test_acc, ada_test_acc, grad_test_acc, svcl_test_acc, svcp_test_acc, svcr_test_acc, stack_test_acc],
    'Training F1 Score': [log_train_f1, dec_train_f1, knn_train_f1, rnd_train_f1, bag_train_f1, ada_train_f1, grad_train_f1, svcl_train_f1, svcp_train_f1, svcr_train_f1, stack_train_f1],
    'Testing F1 Score': [log_test_f1, dec_test_f1, knn_test_f1, rnd_test_f1, bag_test_f1, ada_test_f1, grad_test_f1, svcl_test_f1, svcp_test_f1, svcr_test_f1, stack_test_f1],
    'Training Precision': [log_train_prec, dec_train_prec, knn_train_prec, rnd_train_prec, bag_train_prec, ada_train_prec, grad_train_prec, svcl_train_prec, svcp_train_prec, svcr_train_prec, stack_train_prec],
    'Testing Precision': [log_test_prec, dec_test_prec, knn_test_prec, rnd_test_prec, bag_test_prec, ada_test_prec, grad_test_prec, svcl_test_prec, svcp_test_prec, svcr_test_prec, stack_test_prec],
    'Training ROC-AUC Score': [log_train_roc, dec_train_roc, knn_train_roc, rnd_train_roc, bag_train_roc, ada_train_roc, grad_train_roc, svcl_train_roc, svcp_train_roc, svcr_train_roc, stack_train_roc],
    'Testing ROC-AUC Score': [log_test_roc, dec_test_roc, knn_test_roc, rnd_test_roc, bag_test_roc, ada_test_roc, grad_test_roc, svcl_test_roc, svcp_test_roc, svcr_test_roc, stack_test_roc]
}

results_df = pd.DataFrame(results)
results_df


# In[116]:


results_df.sort_values(by=['Training F1 Score', 'Testing F1 Score'], ascending=False)


# In[117]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)


# In[118]:


#Random Forest Classifier Model
rf_classifier = RandomForestClassifier(max_depth=5)
rf_classifier.fit(X_train, y_train)


# In[119]:


# Training and Testing the model and Measuring Accuracies
y_pred_train = rf_classifier.predict(X_train)
y_pred_test = rf_classifier.predict(X_test)


# In[120]:


import pickle
from pickle import dump,load

# Save the model
file_name = 'rf_classifier.pkl'
model_file = open(file_name, 'wb')
pickle.dump(rf_classifier, model_file)

# Load the model
model_file = open(file_name, 'rb')
loaded_model = pickle.load(model_file)
model_file.close()


# In[121]:


result=loaded_model.score(X_test,y_test)
print(result)


# In[122]:


loaded_model.fit(X,y)
pk=loaded_model.predict(X_test)
pk


# In[ ]:




