import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm

#################Reading of data
class_A = pd.read_csv("q1_classA.csv")
class_B = pd.read_csv("q1_classB.csv")
class_A=class_A.as_matrix()
class_B=class_B.as_matrix()
rows_A=np.shape(class_A)[0]
y_A=np.ones((rows_A,1))
rows_B=np.shape(class_B)[0]
y_B=np.full((rows_B,1),2)
#Visualising classses
plt.plot(class_A[:,0], class_A[:,1], 'ro', label="Class A")
plt.plot(class_B[:,0], class_B[:,1], 'bo', label="Class B")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.show()
y=np.append(y_A,y_B,axis=0)
X=np.append(class_A,class_B,axis=0)
C_values=[0.001, 0.01, 0.1, 1]
y=np.reshape(y,(134,))
#SVM for different C values
for i in range(len(C_values)): 
    clf = svm.SVC(kernel='linear', C=C_values[i])
    clf.fit(X, y)
    
    plt.plot(class_A[:,0], class_A[:,1], 'ro', label="Class A")
    plt.plot(class_B[:,0], class_B[:,1], 'bo', label="Class B")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    # plot the decision function
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
