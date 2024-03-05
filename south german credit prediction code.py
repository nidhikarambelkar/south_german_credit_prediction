# importing the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
#importing dataset
data=pd.read_csv('south german credit risk prediction.csv')
# Renaming columns from German to English
columns_names = ["id", "status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration","installment_rate", "personal_status_sex", "other_debtors", "present_residence", "property","age", "other_installment_plans", "housing", "number_credits", "job", "other_installment_plans", "telephone", "foreign_worker", "credit_risk" ]
data.columns = columns_names
# making copy of dataset
data_df=data.copy()

#droping the index column
data_df = data_df.drop("id", axis=1)
#checking for null values
data_df.isnull().sum()
#no null values found

#Analyzing the data
data_df.describe().transpose() 
data_df.info()
data_corr=data_df.corr()
# Checking target variables counts for balance
data_df["credit_risk"].value_counts()

# Number of unique values in each columns
def unique_value(data_set, column_name):
    return data_set[column_name].nunique()

print("Number of the Unique Values:")
print(unique_value(data_df,list(data_df.columns)))

#making frequency tables
pd.crosstab(data_df["purpose"],data_df["credit_risk"])
pd.crosstab(data_df["property"],data_df["credit_risk"])
pd.crosstab(data_df["foreign_worker"],data_df["credit_risk"])


#Visualisation
#Get the countplot to get the idea about credit_risk 
sns.countplot(x=data_df['credit_risk'], data=data_df)
#There is 300% more postive class than negative class

sns.countplot(x=data_df['purpose'], data=data_df)
#Maximum people have taken loan for furniture/equipment

sns.boxplot(data=data_df, x=data_df["amount"], color='g')
data_df[["amount"]].median()
#Median of amount is 2264.0

sns.pairplot(data_df[["amount", "age","duration"]])
#We can a slight correlation between amount and duration


#LINEAR REGRESSION MODEL
data1=data_df[['duration','amount']]
sns.pairplot(data_df[["amount", "age","duration"]])
#linear regression on duration and amount
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(data1[["duration"]].values.reshape(-1, 1), data1[["amount"]])
y_pred = model.predict(data1["duration"].values.reshape(-1, 1))
residuals = data1[["amount"]] - y_pred
print(residuals)
from sklearn.metrics import r2_score ,accuracy_score
r2_score(data1[["amount"]],y_pred)
#40% of dependent values are predicted by independent values
#Not a good fit for the data


# Histogram
plt.hist(residuals)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()
#Residuals follow normal distribution


#Checking outliers
data1=data_df[['duration','amount']]
data1.corr()
#correlation coefficient is 0.633837 which shows moderate correlation between amount and duration
data1["duration"].value_counts().tail(10)
data1["amount"].value_counts().tail(10)
sns.scatterplot(x=data1["duration"],y=data1["amount"] )
#we can see from the scatterplot there are outliers

#Creating testing and training data
data_df.columns
Y=data_df[['credit_risk']]

X=data_df.drop('credit_risk',axis='columns')
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
#Applying ML algorithms
#naive bayesian
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

nb=BernoulliNB()
nb.fit(X_train,Y_train)
pred=nb.predict(X_test)
accb=accuracy_score(Y_test, pred)
print(accb)
confb=confusion_matrix(Y_test,pred)
sns.heatmap(confb)
classb=classification_report(Y_test,pred)
#Knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
model=KNeighborsClassifier()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
acck=accuracy_score(Y_test, pred)
print(acck)
confk=confusion_matrix(Y_test,pred)
sns.heatmap(confk)
classk=classification_report(Y_test,pred)
#logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
accl=accuracy_score(Y_test, pred)
print(accl)
confl=confusion_matrix(Y_test,pred)
sns.heatmap(confl)
classl=classification_report(Y_test,pred)
#decision trees
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
accd=accuracy_score(Y_test, pred)
print(accd)
confd=confusion_matrix(Y_test,pred)
sns.heatmap(confd)
classd=classification_report(Y_test,pred)
from sklearn import tree
tree.plot_tree(model,filled=True)

#support vector machine
from sklearn.svm import LinearSVC
model=LinearSVC()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
accs=accuracy_score(Y_test, pred)
print(accs)
confs=confusion_matrix(Y_test,pred)
sns.heatmap(confs)
class_s=classification_report(Y_test,pred)

#Comparing all the models
models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()


# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()


# Naive Bayes
from sklearn.naive_bayes import BernoulliNB
models['Naive Bayes'] = BernoulliNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score

accuracy, precision, recall, fpr, tpr, roc_auc = {},{},{},{},{},{}

for key in models.keys():
    models[key].fit(X_train, Y_train)
    predictions = models[key].predict(X_test)
    accuracy[key] = accuracy_score(Y_test, predictions)
    precision[key] = precision_score(Y_test, predictions)
    recall[key] = recall_score(Y_test, predictions)
    fpr[key], tpr[key], _ = roc_curve(Y_test, predictions)
    roc_auc[key] = roc_auc_score(Y_test, predictions)
print(accuracy)
df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()

df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

df_model
#plotting the metrics
ax = df_model.plot.barh()
ax.legend(ncol=len(models.keys()),bbox_to_anchor=(0, 1),loc='lower left',prop={'size': 14})
#We can see that logistic regression has the maximum accuracy


 

