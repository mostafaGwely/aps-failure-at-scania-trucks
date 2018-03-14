import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import  StratifiedKFold
from sklearn.preprocessing import binarize

from time import time 
t1 = time()


#reading the data 
data = pd.read_csv('aps_failure_training_set_processed_8bit.csv')



data["class"][data["class"] > 0] = 1
data["class"][data["class"] < 0] = 0

df = data
df_majority = df[df['class']==0]
df_minority = df[df['class']==1]
# Upsample minority class
from sklearn.utils import resample

df_minority_upsampled = resample(df_minority,replace=True,n_samples=60000,random_state=123)
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

data = df_upsampled



#X (Feature matrix ) Y (response)
X = data.iloc[:,1:]
Y = data.iloc[:,0]
#convert df to matrix 
x,y = X.as_matrix(),Y.as_matrix()


skf = StratifiedKFold(n_splits=5)
predicted =np.zeros(y.shape[0])
y_pred_prob = np.zeros((y.shape[0]))
for train,test in skf.split(x,y):
#    print(train,test)
    x_train = x[train]
    x_test = x[test]
    y_train = y[train]
    y_test = y[test]
    rfc = RandomForestClassifier()

    rfc.fit(x_train, y_train)
    predicted[test] = rfc.predict(x_test)
    y_pred_prob[test] = rfc.predict_proba(x_test)[:,1] #will be used later on tuning the threshold to control Sensitivity and Specificity
    
    
accuracy1 = accuracy_score(y,predicted)
print("accuracy without threshold using RandomForestClassifier ",accuracy1)
confR = confusion_matrix(y,rfc.predict(x))

#thresholding 
accuracy2 = [] 
for i in range(1,100):
    y_pred_class = binarize([y_pred_prob], i/100)[0]
    confR2 = confusion_matrix(y,y_pred_class)
    accuracy2.append((confR2[1,1]+confR2[0,0])/data.shape[0])

for i in range(len(accuracy2)):
    if accuracy2[i] == max(accuracy2):
        print("accuracy with  threshold : ",max(accuracy2)," threshold == ",(i/100))
        y_pred_class = binarize([y_pred_prob], i/100)[0]
        confR2 = confusion_matrix(y,y_pred_class)
        break

print("computational time ",(time()-t1)/60," min")