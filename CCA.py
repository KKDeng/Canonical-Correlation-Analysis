import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd
from numpy.linalg import solve
from sklearn import linear_model
import math

def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    numFeat = len(open(fileName).readline().split('\t'))
    for line in fr.readline():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in numFeat:
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
    return dataMat



def ccaSvd(XArr,YArr,mu1,mu2):
     dataMatX=np.mat(XArr);dataMatY=np.mat(YArr)
     n,p=dataMatX.shape;n,q=dataMatY.shape
     u0=np.random.random((p,1));v0=np.random.random((q,1));k=1
     dataMatXMean = dataMatX - np.ones((n, 1)) * np.mean(dataMatX, axis=0);dataMatYMean = dataMatY - np.ones((n, 1)) * np.mean(dataMatY, axis=0)
     K=np.transpose(dataMatXMean)*dataMatYMean
     for i in range(1,10000):
         u=np.array(K*v0);
         if (norm(u) != 0):
             u = u / norm(u)
         u=np.sign(u) * np.maximum(np.abs(u)-mu1/2,0.0)
         if (norm(u) != 0):
             u = u / norm(u)

         v=np.array(np.transpose(K)*u)
         if (norm(v) != 0):
             v = v / norm(v)
         v = np.sign(v) * np.maximum(np.abs(v) - mu2 / 2,0.0)
         if (norm(v) != 0):
             v = v / norm(v)
         if(norm(v-v0)<0.001):
             break
         v0=v;
         k=k+1
     corr=np.transpose(dataMatXMean*u)*(dataMatYMean*v)
     return u,v,corr,k

def ccaLeas(XArr,YArr,mu1,mu2):
    dataMatX = np.mat(XArr);dataMatY = np.mat(YArr)
    n, p = dataMatX.shape;n, q = dataMatY.shape
    u0 = np.random.random((p, 1));v0 = np.random.random((q, 1));k=1
    dataMatXMean = dataMatX - np.ones((n, 1)) * np.mean(dataMatX, axis=0);dataMatYMean = dataMatY - np.ones((n, 1)) * np.mean(dataMatY, axis=0)
    while(1):
        v = linear_model.Lasso(alpha=mu2/(2*n), tol=0.01).fit(dataMatYMean, dataMatXMean * u0).coef_
        v=np.reshape(v,(q,1))
        v = v / norm(dataMatYMean * v)
        u = linear_model.Lasso(alpha=mu1/(2*n), tol=0.01).fit(dataMatXMean, dataMatYMean * v).coef_
        u=np.reshape(u,(p,1))
        u = u / norm(dataMatXMean * u)
        if (norm(u - u0) < 0.001):
            break
        u0 = u
        k=k+1
    corr = np.transpose(dataMatXMean * u) * (dataMatYMean * v)
    return u,v,corr


def generateData(n,p,q,sigma,cov):
    dataX=[];dataY=[];
    mu=np.random.normal(0,sigma,n)

    u=np.append(np.zeros((np.int(p/2),1)),np.ones((p-np.int(p/2),1)));v=np.append(np.ones((np.int(q/2),1)),np.zeros((q-np.int(q/2),1)))
    covXN=np.int(p/2);covYN=np.int(q/2)
    S=np.zeros((covXN,covXN))
    for i in range(covXN):
        for j in range(covXN):
            S[i,j]=cov ** np.abs(i-j)
    S1 = np.zeros((covYN,covYN))
    for i in range(covYN):
        for j in range(covYN):
            S1[i, j] = cov ** np.abs(i - j)
    Q=np.eye(p);Q1=np.eye(q);
    Q[covXN:p,covXN:p]=S;Q1[0:covYN,0:covYN]=S1
    for i in range(n):
        dataX.append(np.random.multivariate_normal(mu[i]*u,Q,1)[0])
        dataY.append(np.random.multivariate_normal(mu[i]*v,Q1,1)[0])
    return dataX,dataY


def proxNorm(X,lamda):
    X=np.mat(X)
    U, sigma, VT = svd(X)
    svp=sigma[sigma>lamda].size
    if(svp>=1):
        W=sigma[0:svp]-lamda
        Y=U[:,0:svp]* np.diag(W)*VT[0:svp,:]
    else:
        Y=np.zeros(100,100)
    return Y

def ccaLassoSub(X,y,lamda,mu,u):
    n,q=X.shape
    r=15;s=15;t=15;
    J0=np.random.random((n,q));p0=np.random.random((n,1));m0=np.random.random((q,1));
    H0=np.random.random((n,q));gama=np.random.random((n,1));theta=np.random.random((q,1));k=1;
    while(norm(u-m0)>0.002):
        temp=np.dot(X.T,X)
        A=(s+2)*temp+r*np.diag(np.diag(temp))+t*np.eye(q);

        b = np.dot(X.T,-gama+s*p0+2*y)-theta+t*m0+(np.mat(np.diag(np.dot((H0+r*J0).T , X)))).T;
        u=solve(np.array(A),b)

        temp = np.dot(X , np.diag((np.array(u.T))[0])) - 1 / r * H0
        J = proxNorm(temp, lamda / r);

        temp = gama / s + X.dot(u) ;
        if(norm(temp)!=0):
            p=temp/norm(temp)

        temp = u + theta / t;
        m = np.multiply(np.maximum(abs(temp) - mu / t,0), np.sign(temp));

        H = H0 + r * (J - np.dot(X , np.diag((np.array(u.T))[0])));
        gama = gama + s * (X.dot(u)  - p);
        theta = theta + t * (u - m);
        J0 = J;p0 = p;m0 = m;H0 = H;
        k=k+1
    return u





def ccaLasso(xArr, yArr, lamda1, lamda2, mu1, mu2):
    dataMatX = np.mat(xArr);dataMatY = np.mat(yArr)
    n, p = dataMatX.shape;n, q = dataMatY.shape
    u0 = np.random.random((p, 1));v0 = np.random.random((q, 1));k = 1
    dataMatXMean = dataMatX - np.ones((n, 1)) * np.mean(dataMatX, axis=0);dataMatYMean = dataMatY - np.ones((n, 1)) * np.mean(dataMatY, axis=0)
    u0=u0/norm(dataMatXMean*u0);v0=v0/norm(dataMatYMean*v0)
    while(1):
        u=ccaTraceLassoSub(dataMatXMean,dataMatYMean*v0,lamda1,mu1,u0)
        v=ccaTraceLassoSub(dataMatYMean,dataMatXMean*u,lamda2,mu2,v0)
        print(norm(u-u0))
        if(norm(u-u0)<0.02):
            break
        u0=u;v0=v
        k=k+1
    corr = np.transpose(dataMatXMean * u) * (dataMatYMean * v)
    return u, v, corr,k






