import pylab
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as FuncAnimation
import sys
import time

with open(sys.argv[1]) as file1:
    x = np.genfromtxt(file1, delimiter=',')
with open(sys.argv[2]) as file2:
    y = np.genfromtxt(file2, delimiter=',')

x=(x-np.mean(x))/np.std(x)
y=(y-np.mean(y))/np.std(y)
    
m=100
theta0=0
theta1=0
lr=(float)(sys.argv[3])/m
tg=(float)(sys.argv[4])

l0=[]
l1=[]
jthetal=[]

theta0min=0
theta1min=0
theta0max=0
theta1max=0
kk=0
while True:

    ngrad0=(np.sum(y-(theta0 + theta1*x)))
    ngrad1=(np.sum(np.multiply(y-(theta0 + theta1*x), x)))

    theta0=theta0+lr*ngrad0
    theta1=theta1+lr*ngrad1

    if theta0>theta0max:
        theta0max=theta0
    if theta1>theta1max:
        theta1max=theta1
    if theta0<theta0min:
        theta0min=theta0
    if theta1<theta1min:
        theta1min=theta1

    l0.append(theta0)
    l1.append(theta1)
    error= (np.sum((y-(theta0 + theta1*x))**2))/(2.0*m)
    jthetal.append(error)

    kk=kk+1

    if ngrad0<0.0001 and ngrad1<0.0001:
        break

pylab.plot(x,y,'o')
pylab.plot(x,theta0+theta1*x, '-k')
print(theta0, theta1)
pylab.show()
pylab.close()

MN=100


# theta0a=np.linspace(-10,10,MN)
# theta1a=np.linspace(-10,10,MN)

theta0a=np.linspace(theta0min-0.5,theta0max+0.5,MN)
theta1a=np.linspace(theta1min-0.5,theta1max+0.5,MN)
# xx,yy=np.meshgrid(theta0a,theta1a)
theta0a=theta0a.reshape((MN,1))
theta1a=theta1a.reshape((MN,1))

def fun(x1, y1):
  return (np.sum((y-(np.asscalar(x1) + np.asscalar(y1)*x))**2))/(2.0*m)

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax=fig.gca(projection='3d')

X, Y = np.meshgrid(theta0a,theta1a)
zs = np.array([fun(x1,y1) for x1,y1 in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
ax.set_zlabel('J(theta)')

# plt.show()
# print(kk)
for i in range(kk):
    ax.scatter3D(l0[i],l1[i],jthetal[i], 'bo')
    # print(i)
    plt.pause(tg)

plt.show()
# plt.close()

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax=fig.gca(projection='3d')
ax=plt.contour(X,Y,Z)

# ax.set_xlabel('Theta0')
# ax.set_ylabel('Theta1')
# ax.set_zlabel('J(theta)')

# plt.show()
# print(kk)
for i in range(kk):
    plt.plot(l0[i],l1[i], 'bo')
    # print(i)
    plt.pause(tg)

plt.show()

# fig, ax = plt.subplots()

# for i in range(kk):
#     # ax.cla()
#     ax.plot([l0[i]],[l1[i]],[jthetal[i]],c='b',s=[100])
#     # Note that using time.sleep does *not* work here!
#     plt.draw()
#     plt.pause(0.1)
#     time.sleep(tg)

# plt.show()
plt.close()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(theta0a, theta1a, jtheta, 50, cmap='gist_gray')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');
# plt.show()
# print(kk)
# surface plot, live: contours: plt.draw