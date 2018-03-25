import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio



#calculating the sigmoid function


def sigmoid(X,Theta):
    m,n=np.shape(X)
    p,q=np.shape(Theta)
    z=(np.ones((m,p)))/((np.ones((m,p)))+np.exp((-1)*(np.dot(X,Theta.T))))
    return z



#calculating the costfunction 

 
def costfunc(X,y,Theta,lam,num_out,hid_layer):
    m,n=np.shape(X)
    Theta1=np.array([Theta[0:(hid_layer*(n+1)),0]]).T
    Theta2=np.array([Theta[(hid_layer*(n+1)):np.shape(Theta)[0],0]]).T
    Theta1=np.reshape(Theta1,(hid_layer,n+1))
    Theta2=np.reshape(Theta2,(num_out,hid_layer+1))
    a1=np.hstack([np.ones((m,1)),X])
    a2=sigmoid(a1,Theta1)
    a2=np.hstack([np.ones((m,1)),a2])
    a3=sigmoid(a2,Theta2)
    sum0=0
    Y=np.zeros((m,num_out))
    for i in range(1,num_out+1):
        l=np.array(y==i,dtype=np.int64)
        r=np.array([a3[:,i-1]]).T
        sum0=sum0-((np.dot((l.T),np.log(r)))+(np.dot((np.ones((1,m))-(l.T)),(np.log(np.ones((m,1))-r)))))
        Y[:,i-1]=l[:,0]
    t1=np.sum(Theta1**2,axis=0)
    t2=np.sum(Theta2**2,axis=0)
    sum1=np.sum(Theta1**2)+np.sum(Theta2**2)-(t1[0]+t2[0])
    cost=((1/m)*(sum0))+(((lam)/(2*m))*(sum1)) 
    return cost



#implementing backpropagation to calcuate the derivatives


def gradient(X,y,Theta,lam,num_out,hid_layer):
    m,n=np.shape(X)
    Theta1=np.array([Theta[0:(hid_layer*(n+1)),0]]).T
    Theta2=np.array([Theta[(hid_layer*(n+1)):np.shape(Theta)[0],0]]).T
    Theta1=np.reshape(Theta1,(hid_layer,n+1))
    Theta2=np.reshape(Theta2,(num_out,hid_layer+1))
    a1=np.hstack([np.ones((m,1)),X])
    a2=sigmoid(a1,Theta1)
    a2=np.hstack([np.ones((m,1)),a2])
    a3=sigmoid(a2,Theta2)
    Y=np.zeros((m,num_out))
    for i in range(1,num_out+1):
        l=np.array(y==i,dtype=np.int64)
        Y[:,i-1]=l[:,0]
    Del1=np.zeros((hid_layer,n+1))
    Del2=np.zeros((num_out,hid_layer+1))
    for i in range(0,m):
        A1=np.array([a1[i,:]]).T
        A2=np.array([a2[i,:]]).T
        A3=np.array([a3[i,:]]).T
        y0=np.array([Y[i,:]]).T
        del3=A3-y0
        del2=np.dot(Theta2.T,del3)*(A2)*(np.ones((hid_layer+1,1))-A2)
        del2=del2[1:hid_layer+1,0]
        Del1=Del1+np.dot(np.array([del2]).T,A1.T)
        Del2=Del2+np.dot(np.array([del3]).T,A2.T)
    r1=np.hstack([np.zeros((hid_layer,1)),Theta1[:,1:(n+1)]])
    r2=np.hstack([np.zeros((num_out,1)),Theta2[:,1:(hid_layer+1)]])
    Theta1_grad=(1/m)*((Del1)+(lam*(r1)))
    Theta2_grad=(1/m)*((Del2)+(lam*(r2)))
    Theta_grad=np.vstack([np.reshape(Theta1_grad,((np.shape(Theta1_grad)[0]*np.shape(Theta1_grad)[1]),1)),np.reshape(Theta2_grad,(np.shape(Theta2_grad)[0]*np.shape(Theta2_grad)[1]*np.shape(Theta2_grad)[2],1))])
    return Theta_grad



#gradient checking


def checkgrad(X,y,Theta1,Theta2):
    X=X[0:30,0:100]
    y=np.array([y[0:30,0]]).T
    Theta1=Theta1[0:10,0:101]
    Theta2=Theta2[0:5,0:11]
    Theta=np.vstack([np.reshape(Theta1,((np.shape(Theta1)[0]*np.shape(Theta1)[1]),1)),np.reshape(Theta2,((np.shape(Theta2)[0]*np.shape(Theta2)[1]),1))])
    lam=0
    num_out=5
    hid_layer=10
    eps=np.zeros(np.shape(Theta))
    e=0.0001
    Theta_grad=gradient(X,y,Theta,lam,num_out,hid_layer)
    print('Gradient Checking.....')
    for i in range(0,np.shape(Theta_grad)[0]):
        eps[i,0]=e
        cost0=costfunc(X,y,Theta+eps,lam,num_out,hid_layer)
        cost1=costfunc(X,y,Theta-eps,lam,num_out,hid_layer)
        grad=(cost0-cost1)/(2*e)
        print(grad,Theta_grad[i,0])
        eps[i,0]=0
    print('if backpropagation is correct the above 2 values should be almost similar')



#optimizing the parameters

    
def optimpar(X,y,Theta,lam,num_out,hid_layer,alpha,max_iter):
    cost=np.zeros((max_iter,1))
    print('running gradient descent...')

    for i in range(0,max_iter):
        Theta=Theta-alpha* gradient(X,y,Theta,lam,num_out,hid_layer)
        cost[i][0]=costfunc(X,y,Theta,lam,num_out,hid_layer)
        print('iteration',i)
    p=np.array([range(0,max_iter)]).T
    print('The value of cost function after optimizing the parrameters is',cost[max_iter-1][0])
    Dict={'Theta':Theta}
    filemat='ex4weights.mat'
    sio.savemat(filemat,Dict)    #saving the learned parameters to a mat file    
    plt.plot(p,cost)      
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost Function')
    plt.title('The variation of the cost function with the number of iterations')
    plt.show()   
    return Theta



#converting the probabilities into digits from 0 to 9 (0 represents the digit 10)


def condig(X,y,Theta,num_out,hid_layer):
    m,n=np.shape(X)
    Theta1=np.array([Theta[0:(hid_layer*(n+1)),0]]).T
    Theta2=np.array([Theta[(hid_layer*(n+1)):np.shape(Theta)[0],0]]).T
    Theta1=np.reshape(Theta1,(hid_layer,n+1))
    Theta2=np.reshape(Theta2,(num_out,hid_layer+1))
    a1=np.hstack([np.ones((np.shape(X)[0],1)),X])
    a2=sigmoid(a1,Theta1)
    a2=np.hstack([np.ones((np.shape(X)[0],1)),a2])
    a3=sigmoid(a2,Theta2)
    z=np.array([np.max(a3,axis=1)]).T
    z0=np.zeros((np.shape(a3)[0],1),dtype=np.int64)
    for i in range(0,np.shape(a3)[0]):
        for j in range(0,np.shape(a3)[1]):
            if(z[i][0]==a3[i][j]):
                z0[i][0]=j+1
    return z0 

    
    
    


