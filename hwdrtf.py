import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import random



#loading data


filename0='ex4data1.mat'
datadict0=sio.loadmat(filename0)
X=np.array(datadict0['X'])
y=np.array(datadict0['y'])
m,n=np.shape(X)



#structuring the neural network
    

hid_layer=500; #number of hidden units
num_out=10; #number of output units



#dividing the training,cross validation and the test sets


X_train=np.zeros((4000,400))
X_test=np.zeros((1000,400))
y_train=np.zeros((4000,1))
y_test=np.zeros((1000,1))
for i in range(0,10):
    X_train[400*i:400*(i+1),:]=X[500*i:500*i+400,:]
    y_train[400*i:400*(i+1),0]=y[500*i:500*i+400,0]
    X_test[100*i:100*(i+1),:]=X[500*i+400:500*(i+1),:]
    y_test[100*i:100*(i+1),0]=y[500*i+400:500*(i+1),0]
Y_train=np.zeros((4000,num_out))
Y_test=np.zeros((1000,num_out))
for i in range(1,num_out+1):
    l0=np.array(y_train==i,dtype=np.int32)
    Y_train[:,i-1]=l0[:,0]
    l1=np.array(y_test==i,dtype=np.int32)
    Y_test[:,i-1]=l1[:,0]



#randomly initializing the parameters


t0=np.random.random((hid_layer,n))
t1=np.random.random((num_out,hid_layer))
t2=np.random.random((hid_layer,1))
t3=np.random.random((num_out,1))
Theta1=(np.array(t0,dtype=np.float32)*(0.6)-np.ones((hid_layer,n),dtype=np.float32)*0.3).T
Theta2=(np.array(t1,dtype=np.float32)*(0.6)-np.ones((num_out,hid_layer),dtype=np.float32)*0.3).T
Theta1_=(np.array(t2,dtype=np.float32)*(0.6)-np.ones((hid_layer,1),dtype=np.float32)*0.3).T
Theta2_=(np.array(t3,dtype=np.float32)*(0.6)-np.ones((num_out,1),dtype=np.float32)*0.3).T


#defining important parameters and variables to use as tensors


x=tf.placeholder(tf.float32,[None,n])
y0=tf.placeholder(tf.int32,[None,num_out])
w1=tf.Variable(Theta1,tf.float32)
w2=tf.Variable(Theta2,tf.float32)
w10=tf.Variable(Theta1_,tf.float32)
w20=tf.Variable(Theta2_,tf.float32)



#finding the activation values


def activation(x0,w01,w02,w010,w020):
    a1=tf.add(tf.matmul(x0,w01),tf.matmul(tf.ones((tf.shape(x)[0],1)),w010))
    a2=tf.nn.sigmoid(a1)
    a3=tf.nn.sigmoid(tf.add(tf.matmul(a2,w02),tf.matmul(tf.ones((tf.shape(x)[0],1)),w020)))
    return a3



#optimizing the parameters

max_iter=2500
c=np.zeros((max_iter,1))
Y=activation(x,w1,w2,w10,w20)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y,labels=y0))
train=tf.train.AdamOptimizer().minimize(cost)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
costf=np.zeros((max_iter,1))
for i in range(0,max_iter):         
    print('iteration',i+1)
    sess.run(train,{x:X_train,y0:Y_train})
    costf[i][0]=sess.run(cost,{x:X_train,y0:Y_train})
print('The value of cost function after optimizing the parameters is :',costf[max_iter-1,0])
Dict={'w1':w1,'w10':w10,'w2':w2,'w20':w20}
filemat='ex3data1.mat'
sio.savemat(filemat,Dict)    #saving the learned parameters to a mat file    
p=np.array(range(1,max_iter+1))
plt.plot(p,costf)      
plt.xlabel('Number of iterations')
plt.ylabel('Value of Cost Function')
plt.title('The variation of the cost function with the number of iterations')
plt.show()


                             
#calculating the accuracy 

filename1='ex3data1.mat'
Dict=sio.loadmat(filename1)
w1=tf.constant(Dict['w1'],tf.float32)
w1_=tf.constant(Dict['w10'],tf.float32)
w2=tf.constant(Dict['w2'],tf.float32)
w2_=tf.constant(Dict['w20'],tf.float32)
z0=np.array(sess.run(Y,feed_dict={x:X_test,y0:Y_test}))
z=np.array([np.max(z0,axis=1)]).T
pred=np.zeros((np.shape(z0)[0],1),dtype=np.int32)
for i in range(0,np.shape(z0)[0]):
    for j in range(0,np.shape(z0)[1]):
        if(z[i][0]==z0[i][j]):
            pred[i,0]=j+1
pred=tf.Variable(pred,tf.int32)
init=tf.global_variables_initializer()
sess.run(init)
y_tc=tf.placeholder(tf.int32,[None,1])
accuracy=tf.reduce_mean(tf.cast(tf.equal(pred,y_tc),tf.float32))
print('The accuracy on the test set is:',sess.run(accuracy,feed_dict={x:X_test,y0:Y_test,y_tc:y_test}))




#predicting the values with the learned parameters


for i in range(0,10):
    w=random.randint(0,999)
    print('The prediction from the learned parameters is (10 is corresponding to 0)',sess.run(pred[w,0]))
    X=np.reshape(X_test[w,:],(20,20))
    plt.imshow(X,cmap=plt.cm.gray_r,interpolation="nearest")
    plt.show()




                        




                         
                             
                         
                         

                        
                         
                         
                             




                         




                         
                         
                         
                         

            
