import gzip
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


feature_arr_sfs=np.zeros((784,1))
feature_arr_sbs=np.zeros((784,1))
mean_0=np.zeros((784,1))
mean_1=np.zeros((784,1))
mean_2=np.zeros((784,1))
mean_3=np.zeros((784,1))
mean_4=np.zeros((784,1))
mean_5=np.zeros((784,1))
mean_6=np.zeros((784,1))
mean_7=np.zeros((784,1))
mean_8=np.zeros((784,1))
mean_9=np.zeros((784,1))
means_num=[mean_0,mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7,mean_8,mean_9]
mean_tot=np.zeros((784,1))
def obj_fn(mat_1,mat_2): #Objective function calculation
    tot_dist=np.sqrt(np.dot(np.transpose(np.subtract(mat_1,mat_2)),np.subtract(mat_1,mat_2))) #Calculating distance between parameter matrices
    return tot_dist


def nn(x_A,x_T, y_A,y_T): #knn
    dist=cdist(np.transpose(x_A),np.transpose(x_T)) #Calculating distance between each each test point and each training point
    dist_sort=np.argsort(dist, axis=0) #sorting distances
    
    #for k=3
    new_label_3=np.empty([1,10000])
    correct_3=0
    for i in range(10000):
        index1_3=dist_sort[0,i]
        index2_3=dist_sort[1,i]
        index3_3=dist_sort[2,i]
        a=np.array([y_A[0,index1_3],y_A[0,index2_3],y_A[0,index3_3]])
        (values_3,counts_3) = np.unique(a,return_counts=True)
        ind=np.argmax(counts_3) #Choosing the class which is most common among nearest neighbours
        new_label_3[0,i]=values_3[ind]
        if(new_label_3[0,i]==y_T[0,i]): #checking whether the predicted label is right
            correct_3=correct_3+1

    print("accuracy for k=3", correct_3/10000) 


def bidirectional(data_x,data_y,test,test_l):
    global feature_arr_sfs
    global feature_arr_sbs
    m=np.shape(data_x)[1]
    f=np.shape(data_x)[0]
    data_0=np.empty([784,1])
    data_1=np.empty([784,1])
    data_2=np.empty([784,1])
    data_3=np.empty([784,1])
    data_4=np.empty([784,1])
    data_5=np.empty([784,1])
    data_6=np.empty([784,1])
    data_7=np.empty([784,1])
    data_8=np.empty([784,1])
    data_9=np.empty([784,1])    
    datas_num=[data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9]
    for i in range(m):   #Separating datapoints in different matrices based on their class labels
            slice_x=np.reshape(data_x[:,i],(784,1))
            datas_num[int(data_y[0,i])]=np.append(datas_num[int(data_y[0,i])],slice_x, axis=1)

    for i in datas_num:
        i=i[:,1:]


#Calculating means of each class and overall mean which is globally available 
    global means_num, mean_tot
    for i in range(means_num):
        means_num[i]=np.reshape(np.sum(datas_num[i], axis=1)/m,(784,1))
        mean_tot=np.add(mean_tot,means_num[i])
    mean_tot=mean_tot/10
    
    for k in range(f):
        obj_dist_arr=np.zeros([784,1])
        obj_dist_arr_sb=np.full((784,1), np.inf)
        for k_1 in range(f):
            ###############SFS
            if(feature_arr_sfs[k_1,0]!=1 and feature_arr_sbs[k_1,0]!=1): #Checking whether the feature has not been added before and the feature has not been dropped by SBS
                obj_dist_arr[k_1,0]=obj_fn(np.append(mean_0[feature_arr_sfs==True],mean_0[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) + obj_fn(np.append(mean_1[feature_arr_sfs==True],mean_1[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) + obj_fn(np.append(mean_2[feature_arr_sfs==True],mean_2[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) + obj_fn(np.append(mean_3[feature_arr_sfs==True],mean_3[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) +obj_fn(np.append(mean_4[feature_arr_sfs==True],mean_4[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) +obj_fn(np.append(mean_5[feature_arr_sfs==True],mean_5[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) +obj_fn(np.append(mean_6[feature_arr_sfs==True],mean_6[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) +obj_fn(np.append(mean_7[feature_arr_sfs==True],mean_7[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) +obj_fn(np.append(mean_8[feature_arr_sfs==True],mean_8[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0])) +obj_fn(np.append(mean_9[feature_arr_sfs==True],mean_9[k_1,0]),np.append(mean_tot[feature_arr_sfs==True], mean_tot[k_1,0]))          
            else:
                obj_dist_arr[k_1,0]=-1
            
            ###############SBS           
            if(feature_arr_sbs[k_1,0]!=1 and feature_arr_sfs[k_1,0]!=1): #Checking whether the feature has not been dropped before and the feature has not been added by SFS
                pseudo_feature_arr_sbs=np.copy(feature_arr_sbs)
                pseudo_feature_arr_sbs[k_1]=1
                obj_dist_arr_sb[k_1,0]=obj_fn(mean_0[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_1[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_2[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_3[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_4[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_5[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_6[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_7[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_8[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])+obj_fn(mean_9[pseudo_feature_arr_sbs==False],mean_tot[pseudo_feature_arr_sbs==False])
            else:
                obj_dist_arr_sb[k_1,0]=math.inf
                                    
        max_f=np.argmax(obj_dist_arr) #Feature that gives max inter-class distance added
        min_f=np.argmin(obj_dist_arr_sb) #Feature that gives minimum inter-class distance dropped
        if(feature_arr_sbs[max_f,0]==0):
            feature_arr_sfs[max_f,0]=1
        if(feature_arr_sfs[min_f,0]==0):
            feature_arr_sbs[min_f,0]=1


        if(k==9 or k==49 or k==149 or k==391): 
            check_sfs=np.transpose(feature_arr_sfs)[0,:]
            fin_x=data_x[check_sfs==1,:]
            fin_y=data_y
            test_x=test[check_sfs==1,:]
            nn(fin_x,test_x, fin_y,test_l) #Calling knn
            data1 = feature_arr_sfs.reshape( 28, 28, 1)
            image = np.asarray(data1).squeeze()
            plt.imshow(image,cmap='gray') #Feature visualisation
            plt.title('Selected features ' + str(k+1) )
            plt.show()

            
####################Reading training data
ft=gzip.open('t10k-images-idx3-ubyte.gz','r')
lt=gzip.open('t10k-labels-idx1-ubyte.gz','r')
f=gzip.open('train-images-idx3-ubyte.gz','r')
l=gzip.open('train-labels-idx1-ubyte.gz','r')
image_size=28
num_images=60000
x=np.empty([784,60000])
y=np.empty([1,60000])
f.read(16)
l.read(8)
for i in range(num_images):
    buf = f.read(image_size * image_size )
    buf_l=l.read(1)
    data2=np.frombuffer(buf_l, dtype=np.uint8)
    data1 = np.frombuffer(buf, dtype=np.uint8)
    data1 = data1.reshape( image_size, image_size, 1)
    image = np.asarray(data1).squeeze()
    #plt.imshow(image)
    data_slice=np.ravel(image)
    y[:,i]=data2
    x[:,i]=data_slice
    #plt.show()
    if(i==60000):
        break

######################Reading test data
ft.read(16)
lt.read(8)
test=np.empty([784,10000])
test_l=np.empty([1,10000])
for i in range(10000):
    buf_t = ft.read(image_size * image_size )
    buf_t_l= lt.read(1) 
    data2_t= np.frombuffer(buf_t_l, dtype=np.uint8)
    data1_t = np.frombuffer(buf_t, dtype=np.uint8)
    data1_t = data1_t.reshape( image_size, image_size, 1)
    image_t = np.asarray(data1_t).squeeze()
    #plt.imshow(image)
    data_slice_t=np.ravel(image_t)
    test[:,i]=data_slice_t
    test_l[:,i]=data2_t
    #plt.show()
    if(i==10000):
        break

bidirectional(x[:,0:10000],y[:,0:10000],test,test_l) #Calling bidirectional function

