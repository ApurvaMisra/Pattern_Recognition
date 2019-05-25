import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score

######################Reading of data
class_A = pd.read_csv("classA.csv")
class_B = pd.read_csv("classB.csv")
class_A=class_A.as_matrix()
class_B=class_B.as_matrix()
rows_A=np.shape(class_A)[0]
y_A=np.ones((rows_A,1))
rows_B=np.shape(class_B)[0]
y_B=np.full((rows_B,1),2)
#Visualising classes
plt.plot(class_A[:,0], class_A[:,1], 'ro', label="Class A")
plt.plot(class_B[:,0], class_B[:,1], 'bo', label="Class B")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.show()
y=np.append(y_A,y_B,axis=0)
num=np.shape(y)[0]
X=np.append(class_A,class_B,axis=0)
C_values=[0.1, 1, 10, 100]
for i in range(len(C_values)): 
    clf = svm.SVC(kernel='linear', C=C_values[i])    
    data=np.append(X,y, axis=1)
    sum=0    
    #10-times-10-fold cross validation
    for iter in range(10):
        np.random.shuffle(data)
        use_dataframe=data[:,:-1]
        data_y=data[:,-1]
        scores = cross_val_score(clf,use_dataframe , data_y, cv=10)
        sum=sum+scores.mean()        
    print("Accuracy for C=",C_values[i])
    print("is", sum)
    clf.fit(X, y)
    plt.plot(class_A[:,0], class_A[:,1], 'ro', label="Class A")
    plt.plot(class_B[:,0], class_B[:,1], 'bo', label="Class B")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
    plt.title("C="+str(C_values[i]),fontsize=15,color='k')
    plt.show()











