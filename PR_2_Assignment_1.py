from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import random
np.random.seed(9001)
prior1=0.2
prior2=0.3
prior3=0.5
mean1=np.array([3,2])
mean2=np.array([5,4])
mean3=np.array([2,5])
cov1=np.array([[1,-1],[-1,2]])
cov2=np.array([[1,-1],[-1,7]])
cov3=np.array([[0.5,0.5],[0.5,3]])


x1, y1 = np.mgrid[-5:15:0.5 , -5:15:0.5]

x1y1 = np.column_stack([x1.flat, y1.flat])

z1 = multivariate_normal.pdf(x1y1, mean=mean1, cov=cov1) #Generating P(x/C1)
z2 = multivariate_normal.pdf(x1y1, mean=mean2, cov=cov2) #Generating P(x/C2)
z3 = multivariate_normal.pdf(x1y1, mean=mean3, cov=cov3) #Generating P(x/C3)
########for MAP (Priors to be multiplied)
'''
z1=z1*prior1 
z2=z2*prior2
z3=z3*prior3
'''
#####MAP


z1 = z1.reshape(x1.shape)
z2 = z2.reshape(x1.shape)
z3 = z3.reshape(x1.shape)
a=x1y1[:,0]
b=x1y1[:,1]
finz= np.maximum.reduce([z1,z2,z3])
for i in range(40): #Finding the maximum probability among the three classes
    for j in range(40):
        if(z1[i,j]>=z2[i,j] and z1[i,j]>=z3[i,j]):
            finz[i,j]=40
        if(z2[i,j]>=z1[i,j] and z2[i,j]>=z3[i,j]):
            finz[i,j]=150
        if(z3[i,j]>=z2[i,j] and z3[i,j]>=z1[i,j]):
            finz[i,j]=250

plt.scatter(x1,y1,c=finz) #Plotting the boundaries
#plt.title('MAP boundary' )
plt.title('ML boundary' )

plt.plot(3,2,'rx', linewidth=20) #For plotting means
plt.plot(5,4,'rx', linewidth=20)
plt.plot(2,5,'rx', linewidth=20)
plt.contour(x1,y1,z1, levels=1, colors='green', label='Class1') #For plotting contours
plt.contour(x1,y1,z2, levels=1, colors='black', label='Class2')
plt.contour(x1,y1,z3, levels=1, colors='magenta',label='Class3')
plt.legend()


data1=np.random.multivariate_normal(mean1, cov1,600) #Generating smaples based on priors
data2=np.random.multivariate_normal(mean2, cov2,900) #Generating samples based on priors
data3=np.random.multivariate_normal(mean3, cov3,1500) #Generating smaples based on priors
p_err=0
######################Confusion matrix
######################class c1
z1_c1 = multivariate_normal.pdf(data1, mean=mean1, cov=cov1)
z2_c1 = multivariate_normal.pdf(data1, mean=mean2, cov=cov2)
z3_c1 = multivariate_normal.pdf(data1, mean=mean3, cov=cov3)
'''
z1_c1=z1_c1*prior1 #For MAP priors to be multiplied
z2_c1=z2_c1*prior2
z3_c1=z3_c1*prior3
'''
c1_c1=0
c1_c2=0
c1_c3=0
for m_in in range(600):
    if(z1_c1[m_in,]>=z2_c1[m_in,] and z1_c1[m_in,]>=z3_c1[m_in,]):
        c1_c1=c1_c1+1
    if(z2_c1[m_in,]>=z1_c1[m_in,] and z2_c1[m_in,]>=z3_c1[m_in,]):
        c1_c2=c1_c2+1
    if(z3_c1[m_in,]>=z1_c1[m_in,] and z3_c1[m_in,]>=z2_c1[m_in,]):
        c1_c3=c1_c3+1

print("c1 as c1", c1_c1)       
print("c1 as c2", c1_c2) 
print("c1 as c3", c1_c3)


###########################class c2

z1_c2 = multivariate_normal.pdf(data2, mean=mean1, cov=cov1)
z2_c2 = multivariate_normal.pdf(data2, mean=mean2, cov=cov2)
z3_c2 = multivariate_normal.pdf(data2, mean=mean3, cov=cov3)
'''
z1_c2=z1_c2*prior1 #For MAP priors to be multiplied
z2_c2=z2_c2*prior2
z3_c2=z3_c2*prior3
'''
c2_c1=0
c2_c2=0
c2_c3=0
for m_in in range(900):
    if(z1_c2[m_in,]>=z2_c2[m_in,] and z1_c2[m_in,]>=z3_c2[m_in,]):
        c2_c1=c2_c1+1
    if(z2_c2[m_in,]>=z1_c2[m_in,] and z2_c2[m_in,]>=z3_c2[m_in,]):
        c2_c2=c2_c2+1
    if(z3_c2[m_in,]>=z1_c2[m_in,] and z3_c2[m_in,]>=z2_c2[m_in,]):
        c2_c3=c2_c3+1

print("c2 as c1", c2_c1)       
print("c2 as c2", c2_c2) 
print("c2 as c3", c2_c3)



################################class c3

z1_c3 = multivariate_normal.pdf(data3, mean=mean1, cov=cov1)
z2_c3 = multivariate_normal.pdf(data3, mean=mean2, cov=cov2)
z3_c3 = multivariate_normal.pdf(data3, mean=mean3, cov=cov3)
'''
z1_c3=z1_c3*prior1 #For MAP priors to be multiplied
z2_c3=z2_c3*prior2
z3_c3=z3_c3*prior3
'''
c3_c1=0
c3_c2=0
c3_c3=0
for m_in in range(1500):
    if(z1_c3[m_in,]>=z2_c3[m_in,] and z1_c3[m_in,]>=z3_c3[m_in,]):
        c3_c1=c3_c1+1
    if(z2_c3[m_in,]>=z1_c3[m_in,] and z2_c3[m_in,]>=z3_c3[m_in,]):
        c3_c2=c3_c2+1
    if(z3_c3[m_in,]>=z1_c3[m_in,] and z3_c3[m_in,]>=z2_c3[m_in,]):
        c3_c3=c3_c3+1

print("c3 as c1", c3_c1)       
print("c3 as c2", c3_c2) 
print("c3 as c3", c3_c3)


p_err= ((c3_c1+c3_c2)/1500)*prior3 + ((c2_c1+c2_c3)/900)*prior2 + ((c1_c2+c1_c3)/600)*prior1 #Calculating experimental probability of error
print("Prob of error for ML", p_err)
print("shape of z1c1", np.shape(z1_c1)) 


plt.show()