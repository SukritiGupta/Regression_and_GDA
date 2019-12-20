import pylab
import numpy as np
import math
import sys

with open(sys.argv[1]) as file1:
	datax = np.genfromtxt(file1, delimiter=',')
with open(sys.argv[2]) as file2:
	datay = np.genfromtxt(file2, delimiter=',')

Y=datay.T
m=datay.size
ones=np.ones((datax.size/2,1))
X=np.concatenate((ones, datax), axis=1)

theta=np.zeros((3,1))
Y=Y.reshape((m,1))
thetalist=[]
hx=np.reciprocal(ones+np.exp((-1.0)*np.dot(X, theta)))
oldll=-1000000
while True:
	
	deltalt=(np.dot((Y- hx).T,X)).T
	l=[np.asscalar(hx[i][0])*np.asscalar((1.0-hx[i][0])) for i in range(m)]
	B=np.diag(l)
	H=np.dot(X.T,np.dot(B,X))
	theta=theta-np.dot(np.linalg.inv(H),deltalt)
	theta=theta/(math.pow(theta[0]*theta[0]+theta[1]*theta[1]+theta[2]*theta[2],0.5))
	thetalist.append(theta)
	hx=np.reciprocal(ones+np.exp((-1)*np.dot(X, theta)))
	lly0=np.sum(np.log(ones-hx[Y==0]))
	lly1=np.sum(np.log(hx[Y==1]))
	newll=lly1+lly0
	if (newll-oldll)<0.000001:
		break
	oldll=newll


if theta[2]==0:
	top=max(datax[:,1])
	bottom=min(datax[:,1])
	plotx=np.array([(-1)*theta[0]/theta[1], (-1)*theta[0]/theta[1]])
	ploty=np.array([top,bottom])


else:
	left=min(datax[:,0])
	right=max(datax[:,0])
	plotx=np.array([left,right])
	ploty=np.array([(1/theta[2])*(-theta[1]*left-theta[0]),(1/theta[2])*(-theta[1]*right-theta[0])])

label0=datax[datay==0]
label1=datax[datay==1]
# print(theta)

pylab.scatter(label0[:,0], label0[:,1], s=10, label='Zero', c='r', marker='^')
pylab.scatter(label1[:,0], label1[:,1], s=10, label='One', c='c')
pylab.plot(plotx,ploty, '-k')
pylab.show()
# pylab.savefig('b.png')