#---------------------------------Import library-------------------------------
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.grid_search import RandomizedSearchCV
#-----------------------------------Read data----------------------------------
data = pd.read_csv('C:/Users/bjx_3/OneDrive/Documents/mushroom/mushroom_training.csv', sep=',', header=0,
                 error_bad_lines=False, warn_bad_lines=True, low_memory=False)
#-----------------------------------Explore data-------------------------------
print(data.head(2))
#check if there is any null values
print(data.isnull().sum())
#we have two claasification. Either the mushroom is poisonous or edible
print(data.shape)
#----------------------------------Prepare data--------------------------------
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
print(data.head())

#Separating features and label
X = data.iloc[:,0:21]  # input variable
y = data.iloc[:, -1]  # target variable
#--------------------------------Split data-------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape)
# Scale the data to be between -1 and 1
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
#--------------------------Tune model------------------------------------------

svm_model= SVC()
tuned_parameters = {
 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
 #'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
    }

model_svm = RandomizedSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=20)
model_svm.fit(X_train, y_train)
print(model_svm.best_score_)
print(model_svm.grid_scores_)
print(model_svm.best_params_)
#--------------------------Predict test dataset--------------------------------
# fit on test data set
y_pred= model_svm.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix)

# calculate AUC
auc_roc=metrics.classification_report(y_test, y_pred)
print(auc_roc)
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
# plot ROC
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
