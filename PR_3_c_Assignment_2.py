import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

with_scatt=np.zeros((784,784))
b_scatt=np.zeros((784,784))
def scatter(mat): #Calculation of within class scatter
    print(mat)
    global with_scatt
    m=np.shape(mat)[1]
    f=np.shape(mat)[0]
    if(m!=0):
        mean=np.sum(mat, axis=1)/m
        subt=np.transpose(np.subtract(np.transpose(mat),mean))
        scatt=np.dot(subt, np.transpose(subt))
    else:
        scatt=np.zeros((784,784))
    with_scatt+=scatt

def bet_scatt(orig,data): #Calculation of between class scatter
    global b_scatt
    m=np.shape(orig)[1]
    f=np.shape(orig)[0]

    over_mean=np.sum(orig, axis=1)/m
    m_1=np.shape(data)[1]
    f_1=np.shape(data)[0]
    if(m_1!=0):      
        mean=np.sum(data, axis=1)/m_1
        subt=np.subtract(mean,over_mean)
        mult=m_1*(np.dot(subt, np.transpose(subt)))
    else:
        mult=np.zeros((784,784))
    b_scatt+=mult


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

def lda(data_x, data_y, test_x, test_y):
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
    for i in range(m):   #Separating datapoints in different matrices based on their class labels
            slice_x=np.reshape(data_x[:,i],(784,1))
            datas_num[int(data_y[0,i])]=np.append(datas_num[int(data_y[0,i])],slice_x, axis=1) 

    for i in datas_num:
        i=i[:,1:]
    classes=[data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9]
    for i in range(len(classes)): #Calling "within class" scatter function for the 10 class matrices
        scatter(classes[i])

    for i in range(len(classes)): #Calling "between class" scatter function for the 10 class matrices
        bet_scatt(data_x,classes[i])

    obj= np.dot(np.linalg.pinv(with_scatt), b_scatt)
    w,v=np.linalg.eig(obj)#Calculation of eigen values and eigen vectors 
    w_sort=np.argsort(w) #Sorting indices for eigen values
    w_sort=np.flipud(w_sort) #Flipping the sorted matrix
    w_fin=v[:,w_sort[0]]
    w_fin=w_fin.reshape(784,1)
    for vec in range(1,9):
        w_fin=np.c_[ w_fin,v[:,w_sort[vec] ]] #Selecting top 9 eigen vectors for transformation


    nn(np.dot(np.transpose(w_fin), data_x), np.dot(np.transpose(w_fin),test_x),data_y, test_y)#Calling knn on transformed data









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


lda(x[:,0:10000],y[:,0:10000],test, test_l) #Calling LDA function
