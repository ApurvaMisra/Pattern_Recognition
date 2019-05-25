import random
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score

#Adaboost function
def ada_boost(data,test):
    train_num=np.shape(data)[0]
    weights=np.full((train_num,1),(1/train_num))
    result=np.zeros((np.shape(test)[0],50))
    #Loop for 50 SVMs
    for k in range(50):
        clf = svm.SVC(kernel='linear', C=1)
        #Selection of datapoints based on weights
        sel_data=random.choices(population=data, weights=weights,k=100)
        sel_data=np.asarray(sel_data)
        x_data=sel_data[:,:-1]
        y_data=sel_data[:,-1]
        clf.fit(x_data, y_data)
        error=0
        for tot in range(train_num):
            if(clf.predict(data[tot,:-1].reshape(1,-1))!=data[tot,-1]):
                error= error+ weights[tot,:]
        error=error/np.sum(weights)
        if(error<0.5):
            alpha=0.5*math.log(((1-error)/error))
            #Updation of weights
            for iter in range(train_num):
                weights[iter,:]=weights[iter,:]* math.exp(-1*alpha*clf.predict(data[iter,:-1].reshape(1,-1))*data[iter,-1])
            for check in range(np.shape(test)[0]):
                result[check,k]=clf.predict(test[check,:-1].reshape(1,-1))* alpha
        else:
            k=k-1 #If the SVM has an error>0.5, the loop will repeat

    final=np.sum(result,axis=1)
    corr=0
    #checking for accuracy
    for con in range(np.shape(test)[0]):
        if((final[con]<0 and test[con,-1]<0) or (final[con]>0 and test[con,-1]>0)):
            corr=corr+1
    print("Accuracy of each Ensemble",corr/np.shape(test)[0])
    return corr/np.shape(test)[0]
        
################Reading data
class_A = pd.read_csv("classA.csv")
class_B = pd.read_csv("classB.csv")
class_A=class_A.as_matrix()
class_B=class_B.as_matrix()
rows_A=np.shape(class_A)[0]
y_A=np.ones((rows_A,1))
rows_B=np.shape(class_B)[0]
y_B=np.full((rows_B,1),-1)
plt.plot(class_A[:,0], class_A[:,1], 'ro', label="Class A")
plt.plot(class_B[:,0], class_B[:,1], 'bo', label="Class B")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.show()
y=np.append(y_A,y_B,axis=0)
num=np.shape(y)[0]
X=np.append(class_A,class_B,axis=0)
data=np.append(X,y, axis=1)
accuracy=0
accuracy_arr=[]
#10-times-10-fold cross validation
for iter in range(10):
    np.random.shuffle(data)
    for iter_in in range(10):
        rows=np.shape(data)[0]
        slot=int(0.1*rows)
        test_data=data[iter_in*slot:(iter_in*slot)+(slot),:]                
        n1=data[0:iter_in*slot,:]
        n2=data[int(iter_in*slot)+(slot):,:]
        data_new=np.append(n1,n2,axis=0)
        single_acc=ada_boost(data_new,test_data)#Calling Adaboost function
        accuracy=accuracy+single_acc
        accuracy_arr.append(single_acc)  
print("Final accuracy", accuracy/100)
print("Final variance",np.var(accuracy_arr))













