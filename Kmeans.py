import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

K=3 #the number of clusters to be formed


#loading data

filename0='ex7data2.mat'
datadict=sio.loadmat(filename0)
X=np.array(datadict['X'],np.float32)
m,n=np.shape(X)


#randomly choosing the centroids

centroids=np.zeros((K,n),np.float32)
for i in range(0,K):
    w=random.randint(0,m)
    centroids[i,:]=X[w,:]
X=tf.constant(X,tf.float32)    
centroids=tf.Variable(centroids,tf.float32)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)


#forming clusters

while(1):
    centroids_mat=tf.reshape(tf.tile(centroids,[m,1]),[m,K,n])
    X_mat=tf.reshape(tf.tile(X,[1,K]),[m,K,n])
    distances=tf.reduce_sum(tf.square(X_mat-centroids_mat),reduction_indices=2)
    centroids_index=tf.argmin(distances,1)
    total_sum=tf.unsorted_segment_sum(X,centroids_index,K)
    num_total=tf.unsorted_segment_sum(tf.ones_like(X),centroids_index,K)
    c0=sess.run(centroids)
    centroids=total_sum/num_total
    c1=sess.run(centroids)
    l=0
    for i in range(0,K):
        for j in range(0,n):
            if(c1[i][j]==c0[i][j]):
                l=l+1
    if(l==K*n):
       break
Dict={'centroids':centroids,'centroids_index':centroids_index}
filename1='ex7data1.mat'
sio.savemat(filename1,Dict)


#plotting the clusters

idx=np.array(sess.run(centroids_index))
dist=np.array(sess.run(distances))
min_dist=np.zeros((m,1))
X0=np.zeros((m,n))
X=np.array(sess.run(X))
j=0
for i in range(0,m):
    if(idx[i]==0):
       X0[j,:]=X[i,:]
       j=j+1
       l0=j
for i in range(0,m):
    if(idx[i]==1):
       X0[j,:]=X[i,:]
       j=j+1
       l1=j
for i in range(0,m):
    if(idx[i]==2):
       X0[j,:]=X[i,:]
       j=j+1
       l2=j
for i in range(0,m):
    min_dist[i,0]=dist[i,idx[i]]
dist_err=np.sum(min_dist)    
print(dist_err)
plt.scatter(X0[0:l0,0],X0[0:l0,1],color='m',marker='*',s=30)
plt.scatter(X0[l0:l1,0],X0[l0:l1,1],color='r',marker='+',s=30)
plt.scatter(X0[l1:l2,0],X0[l1:l2,1],color='c',s=30)
plt.show()



       
