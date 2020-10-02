import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
titanic = sns.load_dataset('titanic')
#print(titanic.head())

#visualise the count of survivors
sns.countplot(data = titanic, x = titanic['sex']) #There is an overwhelming likelihood that you will die if you are a man. 

#What are the data types?
print(titanic.dtypes)

#Drop the datasets that we do not need
titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'alone', 'adult_male'], axis=1)

#remove the rows with missing values
titanic = titanic.dropna(subset = ['embarked', 'age'])

#We need to get all columns to a number datatype
#Male and Female, 1 and 0 respectively
#Change the embarked cities to numbers too
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
titanic.iloc[:,2] = labelencoder.fit_transform(titanic.iloc[:,2].values)
titanic.iloc[:,7] = labelencoder.fit_transform(titanic.iloc[:,7].values)

#Split the data into independent 'x' and dependent 'y'
X = titanic.iloc[:,1:8].values
Y = titanic.iloc[:,0].values

#Create the training sets
#split the dataset into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#Scale the data (for future prediction)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#create a function with many machine learning models
def models(X_train, Y_train):
    
    #use logistic regresion
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
    
    #Use KNeighbours
    from sklearn.neighbors import KNeighborsClassifier
    kmn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    kmn.fit(X_train, Y_train)
    
    #use SV > linear Kernel
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, Y_train)
    
    #Use SVC > RBF kernel
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, Y_train)
    
    #Use GuassianNB
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)
    
    #Use decision tree classifier
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)
    
    #Use Random forest
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    
    #Print the training accuracy for each model
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Neigbours Regression Training Accuracy:', kmn.score(X_train, Y_train))
    print('[2]SVC Linear Kernel Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]SVC RBF Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]GuassinNB Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Regression Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Regression Training Accuracy:', forest.score(X_train, Y_train))
    
    return log, kmn, svc_lin, svc_rbf, gauss, tree, forest


#Return the final models
model = models(X_train, Y_train)


#Show the confusion matric and accuracy for all of the models on the test data
#This will show that Random Forest Regression Training Accuracy is the best! :)
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    
    #Extract TN, FP, FN, TP
    TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()
    
    test_score = (TP + TN)/ (TP+TN+FN+FP)
    print(cm)
    print('Model[{}] Testing Accuracy = "{}"'.format(i, test_score))


#Get feature importance, age is the most importance for prediction, embarked is the worst
#probably what you would expect!
forest = model[6]
importances = pd.DataFrame({'feature': titanic.iloc[:,1:8].columns, 'importance': np.round(forest.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending =False).set_index('feature')
print(importances)

#Visualise the importnaces
print(importances.plot.bar())


#Print the prediction of the random forest classifier
pred = model[6].predict(X_test)
print(pred)

print()

#print the actual values > What are the differences?
print(Y_test)


#Would you survive? Create a list of your values.
#In this order
pclass        int64
sex           int64
age         float64
sibsp         int64
parch         int64
fare        float64
embarked      int64
my_survival = [[3, 1, 21, 0, 0, 0, 1]]

#You need to scale it like you did with the orginal data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
my_survival_scaled = sc.fit_transform(my_survival)

#print prediction of your survival using random forest classifier (the best model)
pred = model[6].predict(my_survival_scaled)
print(pred)

if pred == 0:
    print('Oh No! You didnt make it')
else:
    print('nice! You survived!')