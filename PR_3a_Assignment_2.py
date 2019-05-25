import gzip
import numpy as np
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
mean_tot=np.zeros((784,1))
def obj_fn(k,k_1):
    intra_dist=0
    #print(check)
    if(check=="sf"):
        mat=mat[1:,:]
    tot_dist_0=np.sqrt(np.dot(np.transpose(np.subtract(mean_0[k_1],mean_tot[k_1])),np.subtract(mean_0[k_1],mean_tot[k_1])))
    tot_dist_1=np.sqrt(np.dot(np.transpose(np.subtract(mean_0[k_1],mean_tot[k_1])),np.subtract(mean_0[k_1],mean_tot[k_1])))
    
    tot_dist=     
    #print("tot dist",tot_dist)    
    return intra_dist


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

    print("accuracy for k=3 and d", correct_3/10000) 











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


    for i in range(m):
        if(data_y[0,i]==0):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_0=np.append(data_0,slice_x, axis=1)
        if(data_y[0,i]==1):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_1=np.append(data_1,slice_x, axis=1)       
        if(data_y[0,i]==2):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_2=np.append(data_2,slice_x, axis=1)
        if(data_y[0,i]==3):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_3=np.append(data_3,slice_x, axis=1)   
        if(data_y[0,i]==4):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_4=np.append(data_4,slice_x, axis=1)
        if(data_y[0,i]==5):
            slice_x=np.reshape(data_x[:,i],(784,1))
            np.append(data_5,slice_x, axis=1)   
        if(data_y[0,i]==6):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_6=np.append(data_6,slice_x, axis=1)
        if(data_y[0,i]==7):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_7=np.append(data_7,slice_x, axis=1)   
        if(data_y[0,i]==8):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_8=np.append(data_8,slice_x, axis=1)
        if(data_y[0,i]==9):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_9=np.append(data_9,slice_x, axis=1)  

    data_0=data_0[:,1:]
    data_1=data_1[:,1:]
    data_2=data_2[:,1:]
    data_3=data_3[:,1:]
    data_4=data_4[:,1:]
    data_5=data_5[:,1:]
    data_6=data_6[:,1:]
    data_7=data_7[:,1:]
    data_8=data_8[:,1:]
    data_9=data_9[:,1:]
    data_0_mat=np.empty([1,np.shape(data_0)[1]])
    data_1_mat=np.empty([1,np.shape(data_1)[1]])
    data_2_mat=np.empty([1,np.shape(data_2)[1]])
    data_3_mat=np.empty([1,np.shape(data_3)[1]])
    data_4_mat=np.empty([1,np.shape(data_4)[1]])
    data_5_mat=np.empty([1,np.shape(data_5)[1]])
    data_6_mat=np.empty([1,np.shape(data_6)[1]])
    data_7_mat=np.empty([1,np.shape(data_7)[1]])
    data_8_mat=np.empty([1,np.shape(data_8)[1]])
    data_9_mat=np.empty([1,np.shape(data_9)[1]])
    


    data_0_mat_sb=data_0
    data_1_mat_sb=data_1
    data_2_mat_sb=data_2
    data_3_mat_sb=data_3
    data_4_mat_sb=data_4
    data_5_mat_sb=data_5
    data_6_mat_sb=data_6
    data_7_mat_sb=data_7
    data_8_mat_sb=data_8
    data_9_mat_sb=data_9 

    mean_0=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_1=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_2=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_3=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_4=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_5=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_6=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_7=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_8=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
    mean_9=np.reshape(np.sum(data_0, axis=1)/m,(784,1))
   
    mean_tot=np.add(mean_tot, mean_0)
    mean_tot=np.add(mean_tot, mean_1)
    mean_tot=np.add(mean_tot, mean_2)
    mean_tot=np.add(mean_tot, mean_3)
    mean_tot=np.add(mean_tot, mean_4)
    mean_tot=np.add(mean_tot, mean_5)
    mean_tot=np.add(mean_tot, mean_6)
    mean_tot=np.add(mean_tot, mean_7)
    mean_tot=np.add(mean_tot, mean_8)
    mean_tot=np.add(mean_tot, mean_9)
    
    


    for k in range(f):
        print("k",k)
        obj_dist_arr=np.zeros([784,1])
        obj_dist_arr_sb=np.zeros([784,1])
        for k_1 in range(f):
            #print("k_1",k_1)
            
            if(feature_arr_sfs[k_1,0]!=1 ):
                obj_dist_arr[k_1,0]=obj_fn(np.append(data_0_mat,np.reshape(data_0[k_1,:],(1,np.shape(data_0)[1])),axis=0),"sf")+obj_fn(np.append(data_1_mat,np.reshape(data_1[k_1,:],(1,np.shape(data_1)[1])),axis=0),"sf")+obj_fn(np.append(data_2_mat,np.reshape(data_2[k_1,:],(1,np.shape(data_2)[1])),axis=0),"sf")+obj_fn(np.append(data_3_mat,np.reshape(data_3[k_1,:],(1,np.shape(data_3)[1])),axis=0),"sf")+obj_fn(np.append(data_4_mat,np.reshape(data_4[k_1,:],(1,np.shape(data_4)[1])),axis=0),"sf")+obj_fn(np.append(data_5_mat,np.reshape(data_5[k_1,:],(1,np.shape(data_5)[1])),axis=0),"sf")+obj_fn(np.append(data_6_mat,np.reshape(data_6[k_1,:],(1,np.shape(data_6)[1])),axis=0),"sf")+obj_fn(np.append(data_7_mat,np.reshape(data_7[k_1,:],(1,np.shape(data_7)[1])),axis=0),"sf")+obj_fn(np.append(data_8_mat,np.reshape(data_8[k_1,:],(1,np.shape(data_8)[1])),axis=0),"sf")+obj_fn(np.append(data_9_mat,np.reshape(data_9[k_1,:],(1,np.shape(data_9)[1])),axis=0),"sf")
                
            else:
                obj_dist_arr[k_1,0]=999999999       
            '''           
            if(feature_arr_sbs[k_1,0]!=1):
                obj_dist_arr_sb[k_1,0]=obj_fn(np.delete(data_0_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_1_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_2_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_3_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_4_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_5_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_6_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_7_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_8_mat_sb, k_1, axis=0),"sb")+obj_fn(np.delete(data_9_mat_sb, k_1, axis=0),"sb")
                max_f=np.argmax(obj_dist_arr_sb)[0]
                #print("max_f",max_f)
                if(feature_arr_sfs[max_f,0]==0):
                    data_0_mat_sb=np.delete(data_0_mat_sb, max_f , axis=0)
                    data_1_mat_sb=np.delete(data_0_mat_sb, max_f, axis=0)
                    data_2_mat_sb=np.delete(data_0_mat_sb, max_f , axis=0)
                    data_3_mat_sb=np.delete(data_0_mat_sb, max_f, axis=0)
                    data_4_mat_sb=np.delete(data_0_mat_sb, max_f, axis=0)
                    data_5_mat_sb=np.delete(data_0_mat_sb, max_f, axis=0)
                    data_6_mat_sb=np.delete(data_0_mat_sb, max_f, axis=0)
                    data_7_mat_sb=np.delete(data_0_mat_sb, max_f, axis=0)
                    data_8_mat_sb=np.delete(data_0_mat_sb, max_f, axis=0)
                    data_9_mat_sb=np.delete(data_0_mat_sb, max_f, axis=0)
                    feature_arr_sbs[max_f,0]=1

              '''      
        max_f=np.argmin(obj_dist_arr)
        print("max for index 1", obj_dist_arr[1,0])
        print("max for index 2", obj_dist_arr[2,0])
        print("max for index 3", obj_dist_arr[3,0])
        print("max_f", max_f)
        print("argmax", np.argmax(obj_dist_arr))
        if(feature_arr_sbs[max_f,0]==0):
            data_0_mat=np.append(data_0_mat,np.reshape(data_0[max_f,:],(1,np.shape(data_0)[1])),axis=0)
            data_1_mat=np.append(data_1_mat,np.reshape(data_1[max_f,:],(1,np.shape(data_1)[1])),axis=0)
            data_2_mat=np.append(data_2_mat,np.reshape(data_2[max_f,:],(1,np.shape(data_2)[1])),axis=0)
            data_3_mat=np.append(data_3_mat,np.reshape(data_3[max_f,:],(1,np.shape(data_3)[1])),axis=0)
            data_4_mat=np.append(data_4_mat,np.reshape(data_4[max_f,:],(1,np.shape(data_4)[1])),axis=0)
            data_5_mat=np.append(data_5_mat,np.reshape(data_5[max_f,:],(1,np.shape(data_5)[1])),axis=0)
            data_6_mat=np.append(data_6_mat,np.reshape(data_6[max_f,:],(1,np.shape(data_6)[1])),axis=0)
            data_7_mat=np.append(data_7_mat,np.reshape(data_7[max_f,:],(1,np.shape(data_7)[1])),axis=0)
            data_8_mat=np.append(data_8_mat,np.reshape(data_8[max_f,:],(1,np.shape(data_8)[1])),axis=0)
            data_9_mat=np.append(data_9_mat,np.reshape(data_9[max_f,:],(1,np.shape(data_9)[1])),axis=0)
            feature_arr_sfs[max_f,0]=1
            print("size of data_0_mat", np.shape(data_0_mat))
        print("sfs feature",np.argwhere(feature_arr_sfs))

'''

        if(k==5):
            check_sfs=np.transpose(feature_arr_sfs)[0,:]
            fin_x=data_x[check_sfs==1,:]
            fin_y=data_y
            test_x=test[check_sfs==1,:]
            print("features",k)
            nn(fin_x,test_x, fin_y,test_l)

        if(k==50):
            check_sfs=np.transpose(feature_arr_sfs)[0,:]
            fin_x=data_x[check_sfs==1,:]
            fin_y=data_y
            test_x=test[check_sfs==1,:]
            print("features",k)
            nn(fin_x,test_x, fin_y,test_l)

        if(k==100):
            check_sfs=np.transpose(feature_arr_sfs)[0,:]
            fin_x=data_x[check_sfs==1,:]
            fin_y=data_y
            test_x=test[check_sfs==1,:]
            print("features",k)
            nn(fin_x,test_x, fin_y,test_l)                    

        if(k==500):
            check_sfs=np.transpose(feature_arr_sfs)[0,:]
            fin_x=data_x[check_sfs==1,:]
            fin_y=data_y
            test_x=test[check_sfs==1,:]
            print("features",k)
            nn(fin_x,test_x, fin_y,test_l)

'''




'''
def sfs(data_x,data_y):
    print("sfs start")
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


    for i in range(m):
        if(data_y[0,i]==0):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_0=np.append(data_0,slice_x, axis=1)
        if(data_y[0,i]==1):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_1=np.append(data_1,slice_x, axis=1)       
        if(data_y[0,i]==2):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_2=np.append(data_2,slice_x, axis=1)
        if(data_y[0,i]==3):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_3=np.append(data_3,slice_x, axis=1)   
        if(data_y[0,i]==4):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_4=np.append(data_4,slice_x, axis=1)
        if(data_y[0,i]==5):
            slice_x=np.reshape(data_x[:,i],(784,1))
            np.append(data_5,slice_x, axis=1)   
        if(data_y[0,i]==6):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_6=np.append(data_6,slice_x, axis=1)
        if(data_y[0,i]==7):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_7=np.append(data_7,slice_x, axis=1)   
        if(data_y[0,i]==8):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_8=np.append(data_8,slice_x, axis=1)
        if(data_y[0,i]==9):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_9=np.append(data_9,slice_x, axis=1)  
        #print("i",i)
    data_0=data_0[:,1:]
    data_1=data_1[:,1:]
    data_2=data_2[:,1:]
    data_3=data_3[:,1:]
    data_4=data_4[:,1:]
    data_5=data_5[:,1:]
    data_6=data_6[:,1:]
    data_7=data_7[:,1:]
    data_8=data_8[:,1:]
    data_9=data_9[:,1:]
    #print("shape of data_5",np.shape(data_5))
    #print("data 5", data_5)
    #print("shape of data_0",np.shape(data_0))
    #print("data 0", data_0[:,0])
    #print("shape of data_1",np.shape(data_1))
    #print("shape of data_6",np.shape(data_6))
    data_0_mat=np.empty([1,np.shape(data_0)[1]])
    data_1_mat=np.empty([1,np.shape(data_1)[1]])
    data_2_mat=np.empty([1,np.shape(data_2)[1]])
    data_3_mat=np.empty([1,np.shape(data_3)[1]])
    data_4_mat=np.empty([1,np.shape(data_4)[1]])
    data_5_mat=np.empty([1,np.shape(data_5)[1]])
    data_6_mat=np.empty([1,np.shape(data_6)[1]])
    data_7_mat=np.empty([1,np.shape(data_7)[1]])
    data_8_mat=np.empty([1,np.shape(data_8)[1]])
    data_9_mat=np.empty([1,np.shape(data_9)[1]])
    
    
    
    for k in range(f):
            obj_dist_arr=np.empty([784,1])
            for k_1 in range(f):
                    print("k_1",k_1)
                    #print("columns in data_0",np.shape(data_0)[1])
                    #print("data_0_mat",np.shape(data_0_mat))
                    #d=np.c_[ d,v[:,i[1]] ]
                    
                    obj_dist_arr[k_1,0]=obj_fn(np.append(data_0_mat,np.reshape(data_0[k_1,:],(1,np.shape(data_0)[1])),axis=0),"sf")+obj_fn(np.append(data_1_mat,np.reshape(data_1[k_1,:],(1,np.shape(data_1)[1])),axis=0),"sf")+obj_fn(np.append(data_2_mat,np.reshape(data_2[k_1,:],(1,np.shape(data_2)[1])),axis=0),"sf")+obj_fn(np.append(data_3_mat,np.reshape(data_3[k_1,:],(1,np.shape(data_3)[1])),axis=0),"sf")+obj_fn(np.append(data_4_mat,np.reshape(data_4[k_1,:],(1,np.shape(data_4)[1])),axis=0),"sf")+obj_fn(np.append(data_5_mat,np.reshape(data_5[k_1,:],(1,np.shape(data_5)[1])),axis=0),"sf")+obj_fn(np.append(data_6_mat,np.reshape(data_6[k_1,:],(1,np.shape(data_6)[1])),axis=0),"sf")+obj_fn(np.append(data_7_mat,np.reshape(data_7[k_1,:],(1,np.shape(data_7)[1])),axis=0),"sf")+obj_fn(np.append(data_8_mat,np.reshape(data_8[k_1,:],(1,np.shape(data_8)[1])),axis=0),"sf")+obj_fn(np.append(data_9_mat,np.reshape(data_9[k_1,:],(1,np.shape(data_9)[1])),axis=0),"sf")
                    max_f=np.argmin(obj_dist_arr)
                    #print("max_f",max_f)
                    if(feature_arr_sbs[max_f]==0):
                        data_0_mat=np.append(data_0_mat,np.reshape(data_0[max_f,:],(1,np.shape(data_0)[1])),axis=0)
                        data_1_mat=np.append(data_1_mat,np.reshape(data_1[max_f,:],(1,np.shape(data_1)[1])),axis=0)
                        data_2_mat=np.append(data_2_mat,np.reshape(data_2[max_f,:],(1,np.shape(data_2)[1])),axis=0)
                        data_3_mat=np.append(data_3_mat,np.reshape(data_3[max_f,:],(1,np.shape(data_3)[1])),axis=0)
                        data_4_mat=np.append(data_4_mat,np.reshape(data_4[max_f,:],(1,np.shape(data_4)[1])),axis=0)
                        data_5_mat=np.append(data_5_mat,np.reshape(data_5[max_f,:],(1,np.shape(data_5)[1])),axis=0)
                        data_6_mat=np.append(data_6_mat,np.reshape(data_6[max_f,:],(1,np.shape(data_6)[1])),axis=0)
                        data_7_mat=np.append(data_7_mat,np.reshape(data_7[max_f,:],(1,np.shape(data_7)[1])),axis=0)
                        data_8_mat=np.append(data_8_mat,np.reshape(data_8[max_f,:],(1,np.shape(data_8)[1])),axis=0)
                        data_9_mat=np.append(data_9_mat,np.reshape(data_9[max_f,:],(1,np.shape(data_9)[1])),axis=0)
                        feature_arr_sfs[max_f]=1
                        control=k
                        sbs(x[:,0:2000],y[:,0:2000])

  '''   
'''
def sbs(data_x,data_y):
    print("sbs start")
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


    for i in range(m):
        if(data_y[0,i]==0):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_0=np.append(data_0,slice_x, axis=1)
        if(data_y[0,i]==1):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_1=np.append(data_1,slice_x, axis=1)       
        if(data_y[0,i]==2):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_2=np.append(data_2,slice_x, axis=1)
        if(data_y[0,i]==3):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_3=np.append(data_3,slice_x, axis=1)   
        if(data_y[0,i]==4):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_4=np.append(data_4,slice_x, axis=1)
        if(data_y[0,i]==5):
            slice_x=np.reshape(data_x[:,i],(784,1))
            np.append(data_5,slice_x, axis=1)   
        if(data_y[0,i]==6):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_6=np.append(data_6,slice_x, axis=1)
        if(data_y[0,i]==7):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_7=np.append(data_7,slice_x, axis=1)   
        if(data_y[0,i]==8):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_8=np.append(data_8,slice_x, axis=1)
        if(data_y[0,i]==9):
            slice_x=np.reshape(data_x[:,i],(784,1))
            data_9=np.append(data_9,slice_x, axis=1)  
        #print("i",i)
    data_0=data_0[:,1:]
    data_1=data_1[:,1:]
    data_2=data_2[:,1:]
    data_3=data_3[:,1:]
    data_4=data_4[:,1:]
    data_5=data_5[:,1:]
    data_6=data_6[:,1:]
    data_7=data_7[:,1:]
    data_8=data_8[:,1:]
    data_9=data_9[:,1:]
    #print("shape of data_5",np.shape(data_5))
    #print("data 5", data_5)
    #print("shape of data_0",np.shape(data_0))
    #print("data 0", data_0[:,0])
    #print("shape of data_1",np.shape(data_1))
    #print("shape of data_6",np.shape(data_6))
    data_0_mat=data_0
    data_1_mat=data_1
    data_2_mat=data_2
    data_3_mat=data_3
    data_4_mat=data_4
    data_5_mat=data_5
    data_6_mat=data_6
    data_7_mat=data_7
    data_8_mat=data_8
    data_9_mat=data_9    
    for k in range(f):
            obj_dist_arr=np.empty([784,1])
            for k_1 in range(f):
                    #print("k_1",k_1)
                    #print("columns in data_0",np.shape(data_0)[1])
                    #print("data_0_mat",np.shape(data_0_mat))
                    #d=np.c_[ d,v[:,i[1]] ]

                    obj_dist_arr[k_1,0]=obj_fn(np.delete(data_0_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_1_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_2_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_3_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_4_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_5_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_6_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_7_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_8_mat, k_1, axis=0),"sb")+obj_fn(np.delete(data_9_mat, k_1, axis=0),"sb")
                    max_f=np.argmin(obj_dist_arr)
                    #print("max_f",max_f)
                    if(feature_arr_sfs[max_f]==0):
                        data_0_mat=np.delete(data_0_mat, max_f , axis=0)
                        data_1_mat=np.delete(data_0_mat, max_f, axis=0)
                        data_2_mat=np.delete(data_0_mat, max_f , axis=0)
                        data_3_mat=np.delete(data_0_mat, max_f, axis=0)
                        data_4_mat=np.delete(data_0_mat, max_f, axis=0)
                        data_5_mat=np.delete(data_0_mat, max_f, axis=0)
                        data_6_mat=np.delete(data_0_mat, max_f, axis=0)
                        data_7_mat=np.delete(data_0_mat, max_f, axis=0)
                        data_8_mat=np.delete(data_0_mat, max_f, axis=0)
                        data_9_mat=np.delete(data_0_mat, max_f, axis=0)
                        feature_arr_sbs[max_f]=1
                        sbs(x[:,0:2000],y[:,0:2000])
                    
    
'''
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
print("shape of x",np.shape(x)) 
print("shape of y",np.shape(y))
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

bidirectional(x[:,0:10000],y[:,0:10000],test,test_l)

print(feature_arr_sbs)
print(feature_arr_sfs)