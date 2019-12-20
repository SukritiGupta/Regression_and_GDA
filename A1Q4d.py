import pylab
import numpy as np
import math
import sys

datax = [map(int,i.strip().split()) for i in open(sys.argv[1]).readlines()]
dt = [i for i in open(sys.argv[2]).readlines()]
m=len(dt)
datay=[(0 if dt[i]=='Alaska\n' else 1) for i in range(m)]
X=np.asarray(datax)
Y1=np.asarray(datay)
Y=Y1.reshape((Y1.size,1))
sum1=sum(Y[Y==1])

phi=1.0*sum1/m
X0=X[np.ix_(Y1==0,)]
X1=X[np.ix_(Y1==1,)]
mu0=1.0*(sum(X0))/(m-sum1)
mu1=1.0*(sum(X1))/sum1
mu0=mu0.reshape((1,2))
mu1=mu1.reshape((1,2))

sigma0=np.zeros((2,2))
sigma1=np.zeros((2,2))

for x in X0:
	sigma0=sigma0+np.dot((x-mu0).T,(x-mu0))

for x in X1:
	sigma1=sigma1+np.dot((x-mu1).T,(x-mu1))

sigma=1.0*(sigma0+sigma1)/m

sigma0=1.0*sigma0/(m-sum1)
sigma1=1.0*sigma1/sum1
# print(phi)
# print(mu0)
# print(mu1)
# print(sigma0)
# print(sigma1)
# print(sigma)

left=min(X[:,0])
right=max(X[:,0])
siginv=np.linalg.inv(sigma)
num=np.dot(mu1,(np.dot(siginv,mu1.T)))-np.dot(mu0,(np.dot(siginv,mu0.T)))- 2*(math.log((phi)/(1-phi)))
num=num/2
mat=np.dot((mu1-mu0),siginv)
lefty=(np.asscalar(num-mat[0][0]*left)/mat[0][1])
righty=np.asscalar((num-mat[0][0]*right)/mat[0][1])
plotx=[left,right]
ploty=[lefty,righty]


siginv0=np.linalg.inv(sigma0)
siginv1=np.linalg.inv(sigma1)
numq=np.dot(mu1,(np.dot(siginv1,mu1.T)))-np.dot(mu0,(np.dot(siginv0,mu0.T)))- 2.0*(math.log((phi)*math.sqrt(np.linalg.det(sigma0))/((1-phi)*math.sqrt(np.linalg.det(sigma1)))))

a1=np.dot(mu0,siginv0)
b1=np.dot(mu1,siginv1)

A=siginv0-siginv1
B=2.0*(b1-a1)
# print(A.shape)
# print(B.shape)
# print(C.shape)


n=1000
diff=1.0*(right-left)/n
plotxq=[]
plotyq=[]
it=np.linspace(left,right,num=n)
for x in it:
	a=(float)(np.asscalar(A[1][1]))
	b=(float)(np.asscalar((1.0*A[0][1]+1.0*A[1][0])*x+1.0*B[0][1]))
	c=(float)(np.asscalar(1.0*B[0][0]*x+1.0*A[0][0]*x*x-numq))
	y=(-b+(math.sqrt(b*b-4.0*a*c)))/(2.0*a)
	plotyq.append(y)
	plotxq.append(x)

plotxq1=[]
plotyq1=[]
for x in it:
	a=(float)(np.asscalar(A[1][1]))
	b=(float)(np.asscalar((1.0*A[0][1]+1.0*A[1][0])*x+1.0*B[0][1]))
	c=(float)(np.asscalar(1.0*B[0][0]*x+1.0*A[0][0]*x*x-numq))
	y=(-b-(math.sqrt(b*b-4.0*a*c)))/(2.0*a)
	plotyq1.append(y)
	plotxq1.append(x)
	
pylab.scatter(X0[:,0], X0[:,1], s=10, label='Alaska', c='r', marker="^")
pylab.scatter(X1[:,0], X1[:,1], s=10, label='Canada', c='b', marker=".")
# pylab.plot(datax,datay,[Y==0]'o')
pylab.plot(plotx,ploty, '-k')
# pylab.plot(plotxq,plotyq)
pylab.plot(plotxq1,plotyq1)
pylab.show()
# pylab.savefig('d.png')