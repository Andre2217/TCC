import numpy as np
import matplotlib.pyplot as plt
from CQC.CQC_EMG import QuadraticClassifier

if __name__ == "__main__":

    X = np.loadtxt("EMG.csv",delimiter=',')
    colors = ['red','blue','magenta','purple','yellow']
    k = 0
    Y = np.empty((0,5))
    for i in range(10):
        
        for j in range(5):
            
            y = -np.ones((1000,5))
            y[:,j] = 1
            Y = np.concatenate((
                Y,y
            ))
            k+=1000

    N,p = X.shape
    c = Y.shape[1]
    for i in range(100):
        seed = np.random.permutation(N)
        Xr = np.copy(X[seed,:])
        Yr = np.copy(Y[seed,:])

        X_treino = Xr[0:int(N*.8),:]
        Y_treino = Yr[0:int(N*.8),:]

        X_teste = Xr[int(N*.8):,:]
        Y_teste = Yr[int(N*.8):,:]

        cq = QuadraticClassifier(X_treino.T,Y_treino.T,5,1)
        cq.fit()
        # x_enviesado = np.zeros((1,2))
        # y_enviesado = -np.ones((1,5))

        # y_enviesado[0,1] = 1
        a,m =cq.test(X_teste.T,Y_teste.T)
        
