import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib as mpl
import random
np.random.seed(9001)
mean1_in=np.array([[0],[0]])
mean2_in=np.array([[0],[0]])
cov1=np.array([[1,0],[0,1]])
cov2=np.array([[1,0.9],[0.9,1]])

data1=np.random.multivariate_normal([0,0], cov1,1000) #Generating smaples
data2=np.random.multivariate_normal([0,0], cov2,1000) #Generating samples
w1, v1 = np.linalg.eig(cov1) #Generating eignen vectors and eigen values
w2, v2 = np.linalg.eig(cov2) #Generating eignen vectors and eigen values
w1_sort=np.argsort(w1) #Sorting the indices for eigen values
w2_sort=np.argsort(w2) #Sorting the indices for eigen values
plt.title('Plot for (a)')
plt.scatter(data1[:,0],data1[:,1])
#plt.Circle(mean1_in, 1)

ax = plt.gca()
ellipse = Ellipse(xy=mean1_in, width=(w1[w1_sort[1]]**0.5)*2, height=(w1[w1_sort[0]]**0.5)*2,angle=np.rad2deg(np.arccos(v1[0,0])),edgecolor='k', fc='None', lw=2)
ax.add_patch(ellipse) # Plotting contour for case(a)

plt.show()
plt.title('Plot for (b)')
plt.scatter(data2[:,0],data2[:,1],color='red')

ax = plt.gca()
ellipse1 = Ellipse(mean2_in, width=(w2[w2_sort[1]]**0.5)*2, height=(w2[w2_sort[0]]**0.5)*2,angle=np.rad2deg(np.arccos(v2[0,0])) ,edgecolor='k', fc='None', lw=2)
# Plotting contour for case(b)
ax.add_patch(ellipse1)

#plt.Circle(mean2_in, 1)
plt.show()
#######Generating covariance matrix for sample data

sample_mean1= np.sum(data1, axis=0)/1000  #Generating sample mean
sample_mean2= np.sum(data2, axis=0)/1000 #Generating sample mean
data1=np.transpose(data1)
data2=np.transpose(data2)
data1_new=np.transpose(np.transpose(data1)-sample_mean1)
data2_new=np.transpose(np.transpose(data2)-sample_mean2)
sample_cov1=np.dot(data1_new,np.transpose(data1_new))/1000 #Generating covariance matrix
sample_cov2=np.dot(data2_new,np.transpose(data2_new))/1000 #Generating covariance matrix
print("Sample cov1",sample_cov1)
print("Sample cov2",sample_cov2)