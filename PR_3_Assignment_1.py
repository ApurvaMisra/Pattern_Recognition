import gzip
import numpy as np
import matplotlib.pyplot as plt
mserr_arr=np.empty([1,784])
derr=np.empty([1,784])
k_index=0
def re_pca(fin_x, d, in_x): #function for reconstruction
    global k_index
    if(np.shape(d)==(784,)):
        print("only once")
        d=np.reshape(d,(784,1))
        fin_x=np.reshape(fin_x,(1,60000))
        
    cons_x=np.dot(d, fin_x) #Calculation of reconstructed matrix
    if(np.shape(d)[1]==1 or  np.shape(d)[1]==10 or  np.shape(d)[1]==50 or  np.shape(d)[1]==250 or np.shape(d)[1]==784):
        data1 = cons_x[:,0].astype(int).reshape( 28,28,1)
        imag = np.asarray(data1.squeeze())
        plt.imshow(imag)
        d_dim=np.shape(d)[1]
        plt.title('Plot for d= ' + str(d_dim) )
        plt.show() #Plot for "5" with varying dimensions
        
    sub_matrix=np.subtract(cons_x,in_x)
    calc=np.sum(np.power(sub_matrix,2))
    mserr=np.divide(calc,(28*28*60000)) #Calculation of mean square error
    x_axis=np.shape(d)
    mserr_arr[0,k_index]=mserr
    derr[0,k_index]=x_axis[1]
    k_index=k_index+1
    plt.plot(x_axis[1],int(mserr), 'ro')
    plt.xlabel('d')
    plt.ylabel('MSE') #Plot of MSE vs d

    


def pca(in_x): #function for pca with pov=95%
    mean=np.sum(in_x, axis=1)/60000
    in_x=np.transpose(np.subtract(np.transpose(in_x),mean))
    cov_mat= np.dot(in_x, np.transpose(in_x))/60000 # calculation of covariance matrix
    w,v=np.linalg.eig(cov_mat)#Calculation of eigen values and eigen vectors for covarianc matrix
    w_sort=np.argsort(w) #Sorting indices for eigen values
    w_sort=np.flipud(w_sort) #Flipping the sorted matrix
    sum_pov=w[w_sort[0]]
    d=np.array(v[:,w_sort[0]]) #Adding the first eigen vector to the transformation matrix d
    wval_sort=np.sort(w) #Eigen values sorted according to values
    wval_sort=np.flipud(wval_sort)
    for ne in range(784):
        if(wval_sort[ne,]!=w[w_sort[ne,],]):
            print("error") #Checking whether the sorted eigen values is same as sorted eigen values through index
        
    plt.plot(w_sort[:,],wval_sort[:,] , 'ro')
    plt.xlabel('d')
    plt.ylabel('Eigen values')
    plt.title("Eigen vs d")
    plt.show() #Plotting eigen values vs d

    for i in enumerate(w_sort[1:,]):
        sum_pov=w[i[1]]+sum_pov
        pov=sum_pov/np.sum(w) #Calculation of POV
        if(pov>=0.95): #Comparison pf POV
            fin_x=np.dot(np.transpose(d),in_x)
            print("Proposed d for POV=", pov)
            print("dvectorshape",np.shape(d))
            break

        d=np.c_[ d,v[:,i[1]]  ]  

    
    return fin_x


def pca_graph(in_x): #function for pca with increasing d
    mean=np.sum(in_x, axis=1)/60000 #mean calculation
    in_x=np.transpose(np.subtract(np.transpose(in_x),mean))
    cov_mat= np.dot(in_x, np.transpose(in_x))/60000 #covariance matrix calculation
    w,v=np.linalg.eig(cov_mat) #eigen vectors and eigen values calcualtion
    w_sort=np.argsort(w) #Sorting of indices based in eigen values
    w_sort=np.flipud(w_sort) #Flipping matrix
    d=np.array(v[:,w_sort[0]])
    d=np.reshape(d,(784,1))
    for i in enumerate(w_sort[1:]):
        fin_x=np.dot(np.transpose(d),in_x)
        re_pca(fin_x,d,in_x)  #Calling reconstruction for varying d
        print("shape of d", np.shape(d))
        
        d=np.c_[ d,v[:,i[1]] ]
        



    
    


f=gzip.open('train-images-idx3-ubyte.gz','r')
l=gzip.open('train-labels-idx1-ubyte.gz','r')

image_size=28
num_images=60000
x=np.empty([784,60000])
f.read(16)
for i in range(num_images): #reading files to put data in matrix
    buf = f.read(image_size * image_size )
    data1 = np.frombuffer(buf, dtype=np.uint8)
    data1 = data1.reshape( image_size, image_size, 1)
    image = np.asarray(data1).squeeze()
    #plt.imshow(image)
    data_slice=np.ravel(image)
    x[:,i]=data_slice
    #plt.show()
    if(i==60000):
        break
    

print("shape of x",np.shape(x))
#trans_x=pca(x) #For pov=95%
pca_graph(x)   #For goinf over varying d

plt.plot(derr, mserr_arr, 'ro')
#plt.xlabel('d')
#plt.ylabel('MSE')
plt.show()


