import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt
import ml   #the ml module module used is also present with this file in this repository
import random

num_out=10 #the number of units in the output layer
hid_layer=450 #the number of units in the hidden layer



#loading data


filename0='ex4data1.mat'
datadict0=sio.loadmat(filename0)
X=np.array(datadict0['X'])
y=np.array(datadict0['y'])
(m,n)=np.shape(X)



#dividing the training,cross validation and the test sets


X_train=np.zeros((4000,400))
X_crossval=np.zeros((500,400))
X_test=np.zeros((500,400))
y_train=np.zeros((4000,1))
y_crossval=np.zeros((500,1))
y_test=np.zeros((500,1))
for i in range(0,10):
    X_train[400*i:400*(i+1),:]=X[500*i:500*i+400,:]
    y_train[400*i:400*(i+1),0]=y[500*i:500*i+400,0]
    X_crossval[50*i:50*(i+1),:]=X[500*i+400:500*i+450,:]
    y_crossval[50*i:50*(i+1),0]=y[500*i+400:500*i+450,0]
    X_test[50*i:50*(i+1),:]=X[500*i+450:500*(i+1),:]
    y_test[50*i:50*(i+1),0]=y[500*i+450:500*(i+1),0]
    
    

#randomly initializing the parameters


Theta1=(np.random.random((hid_layer,n+1))*(0.6))-(np.ones((hid_layer,n+1))*0.3)
Theta2=(np.random.random((num_out,hid_layer+1))*(0.6))-(np.ones((num_out,hid_layer+1))*0.3)
Theta=np.vstack([np.reshape(Theta1,(hid_layer*(n+1),1)),np.reshape(Theta2,(num_out*(hid_layer+1),1))])



#checking the backpropagation implementation


ml.checkgrad(X_train,y_train,Theta1,Theta2)



#optimizing the parameters


alpha=0.1      
lam=0.05
max_iter=2500
Theta=ml.optimpar(X_train,y_train,Theta,lam,num_out,hid_layer,alpha,max_iter)



#verifying the learned parameters on the cross validation and test set


z0=ml.condig(X_crossval,y_crossval,Theta,num_out,hid_layer)
l=0
for i in range(0,np.shape(z0)[0]):
    if(z0[i][0]==y_crossval[i][0]):
        l=l+1
l0=(np.shape(y_crossval)[0]-l)/(np.shape(y_crossval)[0])        
print('The misclassification error on the cross validation set is',l0)


            
#working of handwritten digit recognition on the test set


z1=ml.condig(X_test,y_test,Theta,num_out,hid_layer)
l=0
for i in range(0,np.shape(z0)[0]):
    if(z1[i][0]==y_test[i][0]):
        l=l+1
l0=(np.shape(y_test)[0]-l)/(np.shape(y_test)[0])        
print('The misclassification error on the test set is',l0)
for i in range(0,10):
    w=random.randint(0,499)
    if(z0[w][0]==10):
        z0[w][0]=0
    print('The prediction from the learned parameters is',z0[w][0])
    X=np.reshape(X_test[w,:],(20,20))
    plt.imshow(X,cmap=plt.cm.gray_r,interpolation="nearest")
    plt.show()




        
    
    




